
class DataflowHelper:
    @staticmethod
    def _normalize_handlers(handler: List[str] | str | None) -> List[str] | None:
        if not isinstance(handler, list):
            handler = [handler]
        if handler == ["auto"] or handler == [None]:
            handler = None
        return handler

    @staticmethod
    def _guess_handlers(
        project_name: str | None = None,
        previous_step_names: List[str] | None = None,
        current_step_name: None = None,
        input_locations: List[str] | None = None,
        output_location: None = None,
        mlflow_tracking_uri: str | None = None,
    ) -> Tuple[List[DataTracker | str], DataTracker | str]:
        # override auto-discovery if input or output location is given
        if input_locations is not None and previous_step_names is not None:
            logger.warning(f"Overriding input using {input_locations=}")
            previous_step_names = None

        if output_location is not None and current_step_name is not None:
            logger.warning(f"Overriding input using {output_location=}")
            current_step_name = None

        # consistency check
        ## make sure input is non ambiguous
        if previous_step_names is None and input_locations is None:
            raise ValueError(
                "None of previous_step_names and input_locations are provided: give some input !"
            )
        if previous_step_names is not None and input_locations is not None:
            raise ValueError(
                "Both previous_step_names and input_locations are provided: cannot deal with multi source inputs"
            )
        ## make sure output is non ambiguous
        if current_step_name is None and output_location is None:
            raise ValueError(
                "None of current_step_name and output_location are provided: give some output !"
            )
        if current_step_name is not None and output_location is not None:
            raise ValueError(
                "Both current_step_name and output_location are provided: cannot deal with multi source output"
            )

        # now chose appropriate handler
        ## for input
        if previous_step_names is None:
            input_handler = input_locations
        else:
            if project_name is None:
                raise ValueError("To use a Datatracker, provide a project_name")
            input_handler = [
                DataTracker(project_name, step, mlflow_tracking_uri)
                for step in previous_step_names
            ]
        ## for output
        if current_step_name is None:
            output_handler = output_location[0]
        else:
            if project_name is None:
                raise ValueError("To use a Datatracker, provide a project_name")
            output_handler = DataTracker(
                project_name, current_step_name[0], mlflow_tracking_uri
            )

        return input_handler, output_handler

    def __init__(
        self,
        project_name: str | None = None,
        previous_step_names: str | List[str] | None = None,
        current_step_name: str | None = None,
        input_locations: str | List[str] | None = None,
        output_location: str | None = None,
        mlflow_tracking_uri: str | None = None,
        suffix: str | None = None,
        params: dict | None = None,
        metrics: dict | None = None,
        artifacts: list | None = None,
        s3_config: dict | None = None,
        subfolder: str | None = None,
        lineage: Any | None = None,
        discard_partial_input: bool = True,
        discard_failed_input: bool = False,
        discard_ongoing_runs: bool = True,
        do_push_on_mlflow: bool = True,
    ):
        # normalize every io
        previous_step_names = DataFlowHelper._normalize_handlers(previous_step_names)
        input_locations = DataFlowHelper._normalize_handlers(input_locations)
        current_step_name = DataFlowHelper._normalize_handlers(current_step_name)
        output_location = DataFlowHelper._normalize_handlers(output_location)
        _input_handlers, _output_handler = DataFlowHelper._guess_handlers(
            project_name,
            previous_step_names,
            current_step_name,
            input_locations,
            output_location,
            mlflow_tracking_uri,
        )

        # am I linked to MLFlow ?
        self.push_on_mlflow = (
            isinstance(_output_handler, DataTracker) and do_push_on_mlflow
        )

        # init
        self.project_name = project_name
        self.previous_step_names = previous_step_names
        self.current_step_name = current_step_name
        self._input_handlers = _input_handlers
        self._input_locations: List[str] | None = None  # to be setup later
        self._output_handler = _output_handler
        self.output_location: List[str] | None = None  # to be setup later
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.suffix = suffix
        self.params = params if params else {}
        self.metrics = metrics if metrics else {}
        self.artifacts = artifacts if artifacts else []
        self.s3_config = s3_config
        self.subfolder = subfolder
        self.lineage = lineage
        self.discard_partial_input = discard_partial_input
        self.discard_failed_input = discard_failed_input
        self.discard_ongoing_runs = discard_ongoing_runs

        self.run_id: str | None = None
        self.parent_runs: dict | None = None

        # auto-log some stuff
        if suffix:
            self.params.update({"suffix": suffix})
        if subfolder:
            self.params.update({"subfolder": subfolder})
        if lineage:
            self.params.update({"lineage": lineage})

    @property
    def input_locations(self):
        if self._input_locations is None:
            return None
        elif len(self._input_locations) == 1:
            return self._input_locations[0]
        else:
            return self._input_locations

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if exc_type is None and exc_value is None and traceback is None:
            self.stop()
        else:
            mlflow.end_run("FAILED")
            logger.error("Error occured within context ! Not logging anything")
            print(exc_type)
            print(traceback)

    def start(self):
        # Start by preparing the output to start an experiment
        if isinstance(self._output_handler, str):
            output_location = self._output_handler
        elif isinstance(self._output_handler, DataTracker):
            output_location = self._output_handler.get_new_output_location(
                suffix=self.suffix, metadata=self.params, subfolder=self.subfolder
            )
            run = self._output_handler.start_run()
            self.run_id = run.info.run_id
        # assign final output location
        self.output_location = output_location
        logger.debug(f"Dataflow plugged output dataset {output_location}")

        # deal with inputs
        input_locations = []
        ext_run_id = 0
        for input_loc in self._input_handlers:
            if isinstance(input_loc, str):
                # this was a overriden input: just keep it
                input_locations.append(input_loc)
            elif isinstance(input_loc, DataTracker):
                parent_run = input_loc.get_last_run(
                    lineage=self.lineage,
                    discard_partial=self.discard_partial_input,
                    discard_failed=self.discard_failed_input,
                    discard_ongoing_runs=self.discard_ongoing_runs,
                )
                input_locations.append(parent_run["data"]["params"][OUTPUT_DIR_KEY])
                ext_run_id += 1
                # log things
                self.log_params(
                    {
                        f"depends_no{ext_run_id}_run_id": parent_run["info"]["run_id"],
                        f"depends_no{ext_run_id}_run_url": run_info_to_url(parent_run),
                    }
                )
        # assign final input loc
        self._input_locations = input_locations
        logger.debug(f"Dataflow plugged input dataset {input_locations}")

        if self.push_on_mlflow and isinstance(self._output_handler, DataTracker):
            # start mlflow run and immediatly push metadata to MLFlow
            self._output_handler.push_data_informations(
                input_locations=self.input_locations,  # type: ignore
                output_location=self.output_location,  # type: ignore
                params=self.params,
                metrics=self.metrics,
                artifacts=self.artifacts,
                lineage=self.lineage,
                auto_end_run=False,  # keep the run open
            )

    def stop(self):
        # maybe close the run if MLFlow is in use
        if self.push_on_mlflow:
            mlflow.end_run()
        # maybe write metadata and metrics to S3
        if self.s3_config:
            fs = get_s3_fs_from_config(self.s3_config)
            if self.params:
                metadata_path = osp.join(self.output_location, "metadata.json")  # type: ignore
                with fs.open(metadata_path, "w") as f:
                    json.dump(self.params, f, indent=2)
            if self.metrics:
                metrics_path = osp.join(self.output_location, "metrics.json")  # type: ignore
                with fs.open(metrics_path, "w") as f:
                    json.dump(self.metrics, f, indent=2)
            if self.artifacts:
                logger.info(
                    "You have artifacts but I won't cp them to S3 coz' MLFlow will"
                )
            logger.info(f"Wrote metadata on id={self.run_id}")

    def log_params(self, params: dict | None = None, **kwargs):
        """log_params is a proxy to MLFlow's current run's log_params

        Parameters
        ----------
        metadata : dict | None, optional
            params to log, by default None
        """
        # gather params to log and clean them (MLflow only tolerate keys with alphanumerics, underscores (_), dashes (-), periods (.), spaces (), and slashes (/).)
        params = params if params else {}
        params.update(kwargs)
        params = flatten_dict(params)
        params = {slugify_for_mlflow(k): v for k, v in params.items()}
        # push them directly to MLFlow
        if self.push_on_mlflow:
            mlflow.log_params(params)
        # update self.params with these new params
        self.params.update(params)

    def log_metrics(
        self, metrics: dict | None = None, step: int | None = None, **kwargs
    ):
        # gather metrics to log and clean them (MLflow only tolerate keys with alphanumerics, underscores (_), dashes (-), periods (.), spaces (), and slashes (/).)
        metrics = metrics if metrics else {}
        metrics.update(kwargs)
        metrics = flatten_dict(metrics)
        metrics = {slugify_for_mlflow(k): v for k, v in metrics.items()}
        # push them directly to MLFlow
        if self.push_on_mlflow:
            logger.info("Logging metadata to MLFlow right now")
            mlflow.log_metrics(metrics, step=step)
        # update self.metrics with these new metrics
        if step:
            for metric, value in metrics.items():
                if metric in self.metrics:
                    try:
                        self.metrics[metric].update({step: value})
                    except AttributeError:
                        logger.warning(
                            f"{metric=} was previously logged without step and now it is! Trying to tidy-up ..."
                        )
                        prev_value = self.metrics.pop(metric)
                        self.metrics[metric] = {"no_step": prev_value, step: value}
                else:
                    self.metrics[metric] = {step: value}
        else:
            self.metrics.update(metrics)


    def log_model_tag(self, registered_model_name: str):
        client = MlflowClient(mlflow.get_tracking_uri())
        model_info = client.get_latest_versions(registered_model_name)[0]
        client.set_model_version_tag(
            name=registered_model_name,
            version=model_info.version,
            key=LINEAGE_TAG,
            value=self.lineage,
        )

    def log_artifacts(self, artifacts_local_path: str | List[str]):
        """log_artifacts pushed artifacts to MLFlow and keep track of them

        Parameters
        ----------
        artifacts_local_path : str | List[str]
            string representing a path to a directory or a file to be logger OR a list
            of the same kind of items
        """
        if isinstance(artifacts_local_path, str):
            artifacts_local_path = [artifacts_local_path]

        for artifact in artifacts_local_path:
            if osp.isfile(artifact):
                mlflow.log_artifact(artifact)
                self.artifacts.append(artifact)
            elif osp.isdir(artifact):
                mlflow.log_artifacts(artifact)
                self.artifacts.append(artifact)
            else:
                logger.warning(
                    f"expected str representing path or dir but found {type(artifact_local_path)}: {artifact_local_path}"
                )

    def get_artifact_uri(self, artifact_path: str | None = None) -> str:
        return mlflow.get_artifact_uri(artifact_path)


def run_info_to_url(run_info: dict | mlflow.ActiveRun) -> str:
    if isinstance(run_info, mlflow.ActiveRun):
        run_info = run_info.to_dictionary()
    xp_id = run_info["info"]["experiment_id"]
    run_id = run_info["info"]["run_id"]
    return craft_mlflow_url(xp_id=xp_id, run_id=run_id)


def craft_mlflow_url(xp_id: str, run_id: str | None = None) -> str:
    url = f"{mlflow.get_tracking_uri()}/#/experiments/{xp_id}"
    if run_id:
        url += f"/runs/{run_id}"
    return url