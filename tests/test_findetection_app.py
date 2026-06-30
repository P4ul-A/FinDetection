import base64
import json
import tempfile
import threading
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock, patch

import requests
from PIL import Image

import scripts.picture_preprocessing as picture_preprocessing
from findetection_app import (
    FinDetectionApp,
    PipelineConfig,
    PipelineSummary,
    discover_model_options,
    find_model_logo,
    iter_image_files,
    load_model_badge,
    request_detections,
)


def jpeg_base64() -> str:
    buffer = BytesIO()
    with Image.new("RGB", (4, 4), "white") as image:
        image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def pipeline_app() -> FinDetectionApp:
    app = FinDetectionApp.__new__(FinDetectionApp)
    app.stop_event = threading.Event()
    app.post = Mock()
    return app


class FinDetectionEdgeCaseTests(unittest.TestCase):
    def test_detection_ignores_non_jpeg_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jpeg = root / "fin.JPG"
            jpeg.touch()
            (root / "fin.png").touch()
            (root / "fin.tiff").touch()

            self.assertEqual(iter_image_files(root), [jpeg])

    def test_detection_finds_jpegs_in_nested_folders(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "animal_a" / "day_1" / "first.jpeg"
            second = root / "animal_b" / "day_2" / "second.jpg"
            first.parent.mkdir(parents=True)
            second.parent.mkdir(parents=True)
            first.touch()
            second.touch()

            self.assertEqual(iter_image_files(root), sorted((first, second)))

    def test_detection_excludes_output_folder_inside_input_folder(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jpg"
            output = root / "results"
            generated = output / "source_cropped_0.JPG"
            output.mkdir()
            source.touch()
            generated.touch()

            self.assertEqual(iter_image_files(root, exclude_dir=output), [source])

    def test_models_and_logos_are_discovered_independently(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            models_dir = Path(directory)
            model_without_logo = models_dir / "model_orca.pt"
            orphan_logo = models_dir / "logo_risso.png"
            model_without_logo.touch()
            orphan_logo.touch()

            self.assertEqual(
                discover_model_options(models_dir),
                {"Orca": model_without_logo},
            )
            self.assertIsNone(find_model_logo(model_without_logo))

    def test_empty_input_folder_reports_no_jpegs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            app = pipeline_app()
            config = PipelineConfig(root, root.parent / "output", False, True, 75)

            with self.assertRaisesRegex(RuntimeError, "No JPEG images found"):
                app.run_detection(config, root)

    def test_same_input_and_output_folder_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            app = FinDetectionApp.__new__(FinDetectionApp)
            app.input_dir_var = Mock(get=Mock(return_value=str(root)))
            app.output_dir_var = Mock(get=Mock(return_value=str(root)))

            with patch("findetection_app.messagebox.showerror") as showerror:
                self.assertIsNone(app.read_pipeline_config())

            showerror.assert_called_once()
            self.assertEqual(showerror.call_args.args[0], "Choose a separate output folder")

    def test_failed_jpeg_does_not_stop_remaining_detections(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            (root / "corrupt.jpg").write_bytes(b"not a jpeg")
            (root / "valid.jpg").write_bytes(b"jpeg placeholder")
            app = pipeline_app()
            config = PipelineConfig(root, output, False, True, 75)

            def detect(image_path: Path, _base_url: str) -> list[str]:
                if image_path.name == "corrupt.jpg":
                    raise RuntimeError("invalid image")
                return []

            with patch("findetection_app.request_detections", side_effect=detect):
                result = app.run_detection(config, root)

            self.assertEqual(result, (2, 0, 1))

    def test_no_detection_files_are_logged_together_at_the_end(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            first = root / "Right_sides" / "first.jpg"
            second = root / "second.jpg"
            first.parent.mkdir()
            first.touch()
            second.touch()
            app = pipeline_app()
            config = PipelineConfig(root, output, False, True, 75)

            with patch("findetection_app.request_detections", return_value=[]):
                self.assertEqual(app.run_detection(config, root), (2, 0, 0))

            log_messages = [
                call.args[1]
                for call in app.post.call_args_list
                if call.args[0] == "log"
            ]
            self.assertFalse(
                any(message.startswith("Processing ") for message in log_messages)
            )
            self.assertEqual(
                log_messages[-3:],
                [
                    "No fins detected in 2 images:",
                    "- Right_sides/first.jpg",
                    "- second.jpg",
                ],
            )

    def test_detection_request_retries_timeouts_and_server_errors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "fin.jpg"
            image_path.touch()

            for failure in (
                requests.Timeout("server timed out"),
                Mock(status_code=503, text="unavailable"),
            ):
                with self.subTest(failure=type(failure).__name__):
                    with (
                        patch("findetection_app.requests.post", side_effect=[failure] * 3) as post,
                        patch("findetection_app.time.sleep"),
                    ):
                        with self.assertRaises(RuntimeError):
                            request_detections(image_path, "http://localhost/api")
                    self.assertEqual(post.call_count, 3)

    def test_detection_accepts_both_server_response_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "fin.jpg"
            image_path.touch()

            for field in ("croppedImages", "extractedImages"):
                with self.subTest(field=field):
                    response = Mock(
                        status_code=200,
                        text=json.dumps({"response": {field: ["encoded-image"]}}),
                    )
                    with patch("findetection_app.requests.post", return_value=response):
                        self.assertEqual(
                            request_detections(image_path, "http://localhost/api"),
                            ["encoded-image"],
                        )

    def test_multiple_fins_get_unique_output_names(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            (root / "fin.jpg").touch()
            app = pipeline_app()
            config = PipelineConfig(root, output, False, True, 75)
            encoded_image = jpeg_base64()

            with patch(
                "findetection_app.request_detections",
                return_value=[encoded_image, encoded_image],
            ):
                self.assertEqual(app.run_detection(config, root), (1, 2, 0))

            self.assertTrue((output / "fin_cropped_0.JPG").is_file())
            self.assertTrue((output / "fin_cropped_1.JPG").is_file())

    def test_duplicate_names_in_nested_folders_preserve_structure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            for folder in ("animal_a", "animal_b"):
                source = root / folder / "fin.jpg"
                source.parent.mkdir()
                source.touch()
            app = pipeline_app()
            config = PipelineConfig(root, output, False, True, 75)

            with patch(
                "findetection_app.request_detections",
                return_value=[jpeg_base64()],
            ):
                self.assertEqual(app.run_detection(config, root), (2, 2, 0))

            self.assertTrue((output / "animal_a" / "fin_cropped_0.JPG").is_file())
            self.assertTrue((output / "animal_b" / "fin_cropped_0.JPG").is_file())

    def test_preprocessing_converts_png_tiff_and_raw_to_jpeg(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            with Image.new("RGBA", (4, 4), "red") as image:
                image.save(root / "photo.png")
            with Image.new("RGB", (4, 4), "blue") as image:
                image.save(root / "scan.tiff")
            (root / "camera.cr2").write_bytes(b"raw placeholder")
            app = pipeline_app()
            config = PipelineConfig(root, output, True, False, 80)

            def save_test_image(source: Path, target: Path, quality: int) -> None:
                if source.suffix.lower() == ".cr2":
                    with Image.new("RGB", (4, 4), "green") as image:
                        image.save(target, format="JPEG", quality=quality)
                else:
                    picture_preprocessing.save_as_jpeg(source, target, quality)

            with (
                patch("findetection_app.rawpy", object()),
                patch("findetection_app.save_as_jpeg", side_effect=save_test_image),
            ):
                converted, failed, _, preprocessed_dir, _ = app.run_preprocessing(config)

            self.assertEqual((converted, failed), (3, 0))
            self.assertEqual(preprocessed_dir, output)
            for filename in ("photo.jpg", "scan.jpg", "camera.jpg"):
                self.assertTrue((output / filename).is_file())

    def test_raw_input_without_rawpy_has_actionable_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            (root / "camera.cr2").touch()
            app = pipeline_app()
            config = PipelineConfig(root, output, True, False, 75)

            with patch("findetection_app.rawpy", None):
                with self.assertRaisesRegex(RuntimeError, "rawpy is not installed"):
                    app.run_preprocessing(config)

    def test_cancelled_pipeline_returns_to_reusable_state(self) -> None:
        app = pipeline_app()
        app.pipeline_running = True
        app.stop_event.set()
        app.pipeline_button = Mock()
        app.pipeline_status_var = Mock()
        app.progress_label_var = Mock()

        app.on_pipeline_finished(PipelineSummary(completed=False))

        self.assertFalse(app.pipeline_running)
        self.assertFalse(app.stop_event.is_set())
        app.pipeline_button.configure.assert_called_once_with(
            text="Start pipeline",
            state="normal",
        )
        app.pipeline_status_var.set.assert_called_once_with("Ready")

    def test_png_and_jpeg_model_logos_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            models_dir = Path(directory)
            expected_models: dict[str, Path] = {}
            for identifier, extension in (("orca", ".png"), ("risso", ".jpeg")):
                model_path = models_dir / f"model_{identifier}.pt"
                logo_path = models_dir / f"logo_{identifier}{extension}"
                model_path.touch()
                with Image.new("RGB", (20, 10), "white") as image:
                    image.save(logo_path)
                expected_models[identifier.capitalize()] = model_path

                self.assertEqual(find_model_logo(model_path), logo_path)
                badge = load_model_badge(logo_path)
                self.assertIsNotNone(badge)
                badge.close()

            self.assertEqual(discover_model_options(models_dir), expected_models)

    def test_empty_and_malformed_server_json_do_not_crash_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "fin.jpg"
            image_path.touch()

            empty_response = Mock(status_code=200, text="{}")
            with patch("findetection_app.requests.post", return_value=empty_response):
                self.assertEqual(
                    request_detections(image_path, "http://localhost/api"),
                    [],
                )

            malformed_response = Mock(status_code=200, text="{invalid")
            app = pipeline_app()
            config = PipelineConfig(root, root / "output", False, True, 75)
            with (
                patch("findetection_app.requests.post", return_value=malformed_response),
                patch("findetection_app.time.sleep"),
            ):
                self.assertEqual(app.run_detection(config, root), (1, 0, 1))

    def test_complete_preprocessing_and_detection_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            with Image.new("RGB", (8, 8), "blue") as image:
                image.save(input_dir / "fin.png")
            app = pipeline_app()
            config = PipelineConfig(input_dir, output_dir, True, True, 80)
            response = Mock(
                status_code=200,
                text=json.dumps(
                    {"response": {"croppedImages": [jpeg_base64()]}}
                ),
            )

            with patch("findetection_app.requests.post", return_value=response):
                app.run_pipeline(config)

            self.assertTrue((output_dir / "preprocessed_images" / "fin.jpg").is_file())
            self.assertTrue((output_dir / "fin_cropped_0.JPG").is_file())
            finished = [
                call.args[1]
                for call in app.post.call_args_list
                if call.args[0] == "pipeline_finished"
            ]
            self.assertEqual(len(finished), 1)
            self.assertTrue(finished[0].completed)
            self.assertEqual((finished[0].converted, finished[0].processed), (1, 1))


if __name__ == "__main__":
    unittest.main()
