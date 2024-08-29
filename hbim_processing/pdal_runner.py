import json
import subprocess
import logging
from pathlib import Path
from typing import Any
from jinja2 import Environment, FileSystemLoader


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class PDALPipelineRunner:
    def __init__(self, pipeline_template_path: Path, data_dir: Path | None = Path("./data")):
        self.data_dir = data_dir

        if not pipeline_template_path.is_file():
            raise FileNotFoundError(f"Pipeline template file not found at {pipeline_template_path}")

        self.pipeline_template_path = pipeline_template_path
        template_dir = pipeline_template_path.parent
        self.env = Environment(loader=FileSystemLoader(str(template_dir)))
        self.pipeline = None

    def run(self, parameters: dict[str, Any], delete_original: bool = False) -> Path:
        self.pipeline = self._prepare_pipeline(parameters)
        pipeline_json = json.dumps(self.pipeline)

        command = self.construct_command()

        logger.debug(f"{command=}")
        logger.debug(f"{pipeline_json=}")

        process = subprocess.Popen(
            command.split(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        stdout, stderr = process.communicate(pipeline_json.encode())

        if process.returncode != 0:
            logger.error(f"Error running PDAL pipeline: {stderr.decode()}")
            raise PDALException(f"Error running PDAL pipeline: {stderr.decode()}")

        output_file = self._get_output_file_path(self.pipeline)
        logger.info(f"PDAL pipeline written to: {output_file}")

        if delete_original:
            input_file = self._get_input_file_path(self.pipeline)
            logger.warning(f"Deleting original input file: {input_file}")
            input_file.unlink()

        return output_file

    def _prepare_pipeline(self, parameters: dict[str, Any]) -> list[dict[str, Any]]:
        logger.info("Render pipeline template with parameters")
        template = self.env.get_template(self.pipeline_template_path.name)
        logger.info(f"{parameters=}")
        rendered_template = template.render(parameters)
        logger.debug(f"Rendered template:\n{rendered_template}")
        try:
            rendered_pipeline = json.loads(rendered_template)
            logger.debug(f"Parsed JSON:\n{json.dumps(rendered_pipeline, indent=2)}")
            return rendered_pipeline
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            logger.error(f"Rendered template causing the error:\n{rendered_template}")
            raise

    def _get_output_file_path(self, pipeline: list[dict[str, Any]]) -> Path:
        for stage in pipeline:
            if stage.get("type", "").startswith("writers."):
                return Path(stage["filename"])
        raise PDALException("Output file path could not be determined from the pipeline stages.")

    def _get_input_file_path(self, pipeline: list[dict[str, Any]]) -> Path:
        for stage in pipeline:
            if stage.get("type", "").startswith("readers."):
                return Path(stage["filename"])
        raise PDALException("Input file path could not be determined from the pipeline stages.")

    def construct_command(self) -> str:
        return "pdal pipeline --stdin"

class PDALException(Exception):
    pass
