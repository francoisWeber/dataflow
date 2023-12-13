from typing import Tuple, List
from dataflow.utils.tools import slugify
from loguru import logger
from abc import abstractmethod, abstractclassmethod
from dataflow.utils.exceptions import NoSuchExperimentException
from difflib import get_close_matches
from dataclasses import dataclass

@dataclass
class Experiment:
    name: str
    id: str
        
class Client:
    def __init__(self):
        ...
        
    @abstractmethod
    def get_experiment_by_name(self, xp_name: str) -> dict:
        ...
        
    @abstractmethod
    def search_experiments(self, filter_string: str) -> List[Experiment]:
        ...


PROJECT_STEP_SEPARATOR = ":"

class Tracker1:
    project_step_sep: str = ":"
    client: Client = Client()
    
    def __init__(self, project_name: str, step_name: str, client: Client, resolve_typo: bool = True):
        """__init__ creates a mapper to a kind of dataset

        Parameters
        ----------
        project_name : str, optional
            the project that requires the current data processing, by default None
        step_name : str, optional
            the name of current data processing step within the project, by default None
        """
        self.client = client
        project_name = self._slugify_name(project_name)
        step_name = self._slugify_name(step_name)
        project_name, step_name, xp_name = self._maybe_fix_names(
            project_name, step_name, resolve_typo=resolve_typo
        )
            
        self.project_name = project_name
        self.step_name = step_name
        self.experiment_name = xp_name
        self.experiment = self.client.get_experiment_by_name(self.experiment_name)
        
        
    @classmethod
    def _clean_names(cls, project_name: str, step_name: str, cutoff=0.7) -> Tuple[str]:
        project_name = cls._slugify_name(project_name)
        step_name = cls._slugify_name(step_name)
        # try to fetch the experiment without corrections
        
        
    @classmethod
    def _slugify_name(cls, name: str) -> str:
        if slugify(name) != name:
            name = slugify(name)
            logger.debug(f"Slugified name: {name}")
        return name
    
    @classmethod
    def _get_experiment_name(cls, project_name: str, step_name: str) -> str:
        """_get_experiment_name craft an XP name based on project name + step

        Parameters
        ----------
        project_name : str
            name of your project
        step_name : str
            name of the current processing step

        Returns
        -------
        str
            an experiment name
        """
        return cls.project_step_sep.join([project_name, step_name])
    
    
    def _maybe_fix_names(self, project_name, step_name, cutoff=0.7, resolve_typo: bool=True) -> Tuple[str]:
        xp_name = self._get_experiment_name(project_name, step_name)
        # try to correct the name with the closest possible match
        
        experiment_info = self.client.get_experiment_by_name(xp_name)
        if experiment_info:
            return project_name, step_name, xp_name
        elif resolve_typo:
            # try to find a close xp name
            this_projects_xps = self.client.search_experiments(
                filter_string=f"name LIKE '{project_name}%'"
            )
            possibilities = [xp.name for xp in this_projects_xps]
            matching_names = get_close_matches(
                xp_name,
                possibilities,
                n=1,
                cutoff=cutoff,
            )
            if len(matching_names) > 0:
                fixed_xp_name = matching_names[0]
                logger.warning(
                    f"XP {xp_name} does not exist but {fixed_xp_name} does: trying this one instead!"
                )
                project_name, fixed_name_step = fixed_xp_name.split(
                    PROJECT_STEP_SEPARATOR
                )
                return project_name, fixed_name_step, fixed_xp_name
        else:
            raise NoSuchExperimentException(project_name, step_name)

    def __repr__(self) -> str:
        return f"Tracker({self.experiment_name})"

    def __str__(self) -> str:
        return f"Tracker for project {self.project_name} - step {self.step_name}"
    
    staticmethod
    def create_new_experiment(project_name: str, step_name: str) -> None:
        self.client.
        


# TODO: partir du Tracker actuel > le refacto pour extraire toutes les références à MLFlow dans des méthodes à part > les rendre abstraites

class Tracker:
    """Keeps track of dataset using MLFlow - a bit like DVC with Git"""
    
    client: Client = None

    def __init__(
        self,
        project_name: str,
        step_name: str,
        mlflow_tracking_uri: str | None = None,
    ):
        """__init__ creates a mapper to a kind of dataset

        Parameters
        ----------
        project_name : str, optional
            the project that requires the current data processing, by default None
        step_name : str, optional
            the name of current data processing step within the project, by default None
        mlflow_tracking_uri : str, optional
            The MLFlow endpoint if is not already set thru env var MLFLOW_TRACKING_URI, by default None
        """
        project_name = self._maybe_slugify_name(project_name)
        step_name = self._maybe_slugify_name(step_name)
        project_name, step_name, xp_name = self._maybe_fix_names(
            project_name, step_name
        )
        self.project_name = project_name
        self.step_name = step_name
        self.experiment_name = xp_name
        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.experiment = DataTracker.get_xp_by_name(self.experiment_name)

    def __repr__(self) -> str:
        return f"DataTracker({self.experiment_name})"

    def __str__(self) -> str:
        return f"Data-tracker for project {self.project_name} - step {self.step_name}"

    @staticmethod
    def _maybe_slugify_name(name: str) -> str:
        if slugify(name) != name:
            name = slugify(name)
            logger.debug(f"Slugified name: {name}")
        return name

    @staticmethod
    def _get_experiment_name(
        project_name: str,
        step_name: str,
    ) -> str:
        """_get_experiment_name craft an MLFlow XP name based on project name + step

        Parameters
        ----------
        project_name : str
            name of your project
        step_name : str
            name of the current processing step

        Returns
        -------
        str
            an MLFlow experiment name
        """
        project_name = DataTracker._maybe_slugify_name(project_name)
        step_name = DataTracker._maybe_slugify_name(step_name)
        return PROJECT_STEP_SEPARATOR.join([project_name, step_name])
    
    def get_experiment_by_name(self, xp_name: str):
        return self.client.get_experiment_by_name(xp_name)

    @staticmethod
    def _maybe_fix_names(project_name, step_name, cutoff=0.7) -> Tuple[str]:
        xp_name = DataTracker._get_experiment_name(project_name, step_name)
        # try to correct the name with the closest possible match
        client = mlflow.MlflowClient()
        if client.get_experiment_by_name(xp_name) is not None:
            return project_name, step_name, xp_name
        else:
            this_projects_xps = client.search_experiments(
                filter_string=f"name LIKE '{project_name}%'"
            )
            possibilities = [xp.name for xp in this_projects_xps]
            matching_names = get_close_matches(
                xp_name,
                possibilities,
                n=1,
                cutoff=cutoff,
            )
            if len(matching_names) > 0:
                fixed_xp_name = matching_names[0]
                logger.warning(
                    f"XP {xp_name} does not exist but {fixed_xp_name} does: trying this one instead!"
                )
                project_name, fixed_name_step = fixed_xp_name.split(
                    PROJECT_STEP_SEPARATOR
                )
                return project_name, fixed_name_step, fixed_xp_name
            else:
                raise NoSuchExperimentOnMLFlowException(project_name, step_name)

    @staticmethod
    def create_new_projet_step(
        project_name: str,
        step_name: str,
        project_main_dir: str,
        mlflow_tracking_uri: str | None = None,
    ):
        """create_new_projet_step creates a new xp on MLflow for your project/step

        Parameters
        ----------
        project_name : str
            name of your project
        step_name : str
            name of the new processing step
        project_main_dir : str
            Path of the official main directory of your project
        mlflow_tracking_uri : str | None, optional
            a tracking URI if not available as env var, by default None
        """
        experiment_name = DataTracker._get_experiment_name(
            project_name=project_name, step_name=step_name
        )
        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
        xp_id = mlflow.create_experiment(
            name=experiment_name, tags={MAIN_DIR_TAG: project_main_dir}
        )
        new_xp_url = craft_mlflow_url(xp_id=xp_id)
        logger.info(
            f"Created new experiment for step {step_name} of project {project_name} at {new_xp_url}"
        )

    @staticmethod
    def get_xp_by_name(experiment_name: str) -> Experiment:
        """get_xp_by_name_or_id returns an MLFlow experiment based on an experiment name or id

        Parameters
        ----------
        experiment_name : str, optional
            MLFlow name of an existing experiment to retrieve, by default None

        Returns
        -------
        Experiment
            The MLFlow experiment associated to the ID/name
        """
        logger.info(f"Guessing experiment from {experiment_name=} ...")
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            logger.critical(
                f"{experiment_name=} does not exist on MLFlow: create it with DataTracker.create_new_projet_step"
            )
            raise ValueError(f"Experiment {experiment_name} does not exist")
        return experiment

    def get_latest_runs(
        self,
        n: int,
        discard_failed=False,
        discard_partial=True,
        discard_ongoing_runs=True,
        filters: str | list | None = None,
        lineage: str | None = None,
    ) -> List[dict]:
        """get_latest_runs fetch the latest runs of the project/step

        Parameters
        ----------
        n : int
            number of runs to fetch
        discard_partial: bool
            prevent from fetching a run with a max_collect, thus being a partial dataset, likely a debug dataset
        filters: list or None
            additional "AND" selection filters

        Returns
        -------
        List[dict]
            list of dict description of each n latest runs
        """
        if filters is None:
            filters = []
        if isinstance(filters, str):
            filters = [filters]
        if lineage:
            logger.warning(f"Retaining only runs having {lineage=}")
            filters.append(f'tags.{LINEAGE_TAG} = "{lineage}"')
        if discard_failed:
            logger.warning("Discarding runs that failed")
            filters.append('attributes.status != "FAILED"')
        if discard_ongoing_runs:
            logger.warning("Discarding runs that are ongoing")
            filters.append('attributes.status != "RUNNING"')
        if discard_partial:
            logger.warning("Discarding partial runs with max_collect >= 0")
            filters.append("params.max_collect ILIKE '-1'")

        runs = mlflow.search_runs(
            self.experiment.experiment_id,
            order_by=["start_time DESC"],
            max_results=n,
            output_format="list",
            filter_string=" AND ".join(filters),
            run_view_type=mlflow.entities.ViewType.ALL,  # allow ongoing runs
        )
        runs = [run.to_dictionary() for run in runs]
        if (l := len(runs)) < n:
            logger.error(f"Asked for {n} runs but I found only {l} runs ! :o")
            logger.error(f"Maybe release some of your constraints: {filters}")
        if any([run["info"]["status"] != "FINISHED" for run in runs]):
            logger.warning("Some fetched runs have failed")
            if n == 1:
                logger.warning("Refetching latest FINISHED run")
                return self.get_latest_runs(
                    n, discard_failed=True, discard_partial=discard_partial
                )
        return runs

    def get_last_run(
        self,
        discard_failed=False,
        discard_partial=True,
        discard_ongoing_runs=True,
        lineage: str | None = None,
    ) -> dict:
        """get_latest_run fetch the last run of the project/step

        Returns
        -------
        dict
            dict description of the last run
        """
        try:
            return self.get_latest_runs(
                n=1,
                discard_failed=discard_failed,
                discard_partial=discard_partial,
                discard_ongoing_runs=discard_ongoing_runs,
                lineage=lineage,
            )[0]
        except IndexError as exc:
            logger.error(f"Found not run with {lineage=} :o")
            raise IndexError() from exc

    def get_last_artifact(
        self,
        lineage: str | None = None,
        artifact_name: str | None = None,
        discard_ongoing_runs=True,
        discard_partial: bool = True,
        artifact_type: str | None = None,
    ):
        """
        Download the artifact with the given name from the last run

        Returns
        -------
        Any
            the loaded artifact if txt or json, or the path of the tmp file
        """
        _last_run = self.get_last_run(
            lineage=lineage,
            discard_partial=discard_partial,
            discard_ongoing_runs=discard_ongoing_runs,
        )
        artifact_uri = osp.join(
            _last_run.get("info", {}).get("artifact_uri", None), artifact_name
        )

        if artifact_type is None:
            return mlflow.artifacts.download_artifacts(artifact_uri)

        if artifact_type == "json":
            return mlflow.artifacts.load_dict(artifact_uri)
        elif artifact_type == "txt":
            return mlflow.artifacts.load_text(artifact_uri)
        else:
            raise ValueError(
                f"Artifact type {artifact_type} is unknown, please choose from ['json', 'txt'] or do not pass a type if you want to get a local file."
            )


    def get_latest_output_locations(
        self,
        n: int = 1,
        fail_fast: bool = True,
        discard_failed=False,
        discard_partial=True,
        discard_ongoing_runs=True,
        filters: list | None = None,
        lineage: str | None = None,
    ) -> List[str]:
        """get_latest_output_locations fetch the output directories of the n latest runs

        Parameters
        ----------
        n : int, optional
            number of output locations to fetch, by default 1
        fail_fast : bool, optional
            wether to fail if any output location information is missing, by default True
        filters: list or None
            additional "AND" selection filters

        Returns
        -------
        List[str]
            list of output locations

        """
        runs = self.get_latest_runs(
            n=n,
            discard_failed=discard_failed,
            discard_partial=discard_partial,
            discard_ongoing_runs=discard_ongoing_runs,
            filters=filters,
            lineage=lineage,
        )
        runs_outputs = []
        for run in runs:
            try:
                run_output = run["data"]["params"][OUTPUT_DIR_KEY]
                runs_outputs.append(run_output)
            except KeyError as e:
                run_id = run["info"]["run_id"]
                if fail_fast:
                    logger.critical(
                        f"{run_id} has no key named {OUTPUT_DIR_KEY} which should hold the data location"
                    )
                    raise e
                else:
                    logger.critical(
                        f"Skipping {run_id} that has no key named {OUTPUT_DIR_KEY} ({fail_fast=})"
                    )
        return runs_outputs

    def get_last_output_location(
        self,
        discard_failed=True,
        discard_partial=True,
        discard_ongoing_runs=True,
        filters: List[str] | str | None = None,
        lineage: Any | None = None,
    ) -> str:
        """get_last_output_location fetch the last output location of the project/step

        Returns
        -------
        str
            latest recorded output location
        """
        return self.get_latest_output_locations(
            n=1,
            discard_failed=discard_failed,
            discard_partial=discard_partial,
            discard_ongoing_runs=discard_ongoing_runs,
            lineage=lineage,
            filters=filters,
        )[0]

    def start_run(self) -> mlflow.ActiveRun:
        run = mlflow.start_run(experiment_id=self.experiment.experiment_id)
        run_url = run_info_to_url(run)
        logger.info(f"Started an MLFlow run available at {run_url} !")
        return run

    def push_data_informations(
        self,
        input_locations: str | List[str],
        output_location: str,
        params: dict | None = None,
        metrics: dict | None = None,
        artifacts: list | None = None,
        lineage: str | None = None,
        auto_end_run: bool = True,
    ):
        """push_data_informations push IO information + metadata about processing step

        Parameters
        ----------
        input_location : str
            the source of processed data
        output_location : str
            the output location of the processed data
        params : dict, optional
            some params about the processing itself, by default None
        metrics : dict, optional
            some metrics about the processing itself, by default None
        lineage: str, optional
            If you want to keep track of a perticular chain of steps, provide any data
            making able to distinguish this runs from random ones
        """
        if isinstance(input_locations, str):
            input_locations = [input_locations]
        run = mlflow.active_run()
        if run is None:
            run = self.start_run()

        # avoid context here so we can use the current run somewhere else
        mandatory_params = {OUTPUT_DIR_KEY: output_location}
        for i, input_location in enumerate(input_locations):
            mandatory_params[f"{INPUT_DIR_KEY}-{i}"] = input_location
        mlflow.log_params(mandatory_params)
        if params:
            mlflow.log_params(flatten_dict(params))

        if metrics:
            mlflow.log_metrics(flatten_dict(metrics))

        if artifacts:
            for artifact in artifacts:
                if osp.isfile(artifact):
                    mlflow.log_artifact(artifact)
                elif osp.isdir(artifact):
                    mlflow.log_artifacts(artifact)

        if lineage:
            logger.info(f"Tagging it with lineage {lineage}")
            mlflow.set_tag(LINEAGE_TAG, lineage)

        if auto_end_run:
            mlflow.end_run()
            logger.info(f"Auto-ending this run :)")

    def get_new_output_location(
        self,
        main_directory: str | None = None,
        suffix: str | None = None,
        metadata: dict | None = None,
        subfolder: str | None = None,
    ) -> str:
        """get_new_output_location provide a new output dir following a simple layout

        Parameters
        ----------
        main_directory : str | None, optional
            another main directory than the one registred inside MLF's experiment tag, by default None
        suffix : str | None, optional
            optional suffix for your datageneration, by default None
        metadata : dict | None, optional
            optional metadata digested into a hash, by default None

        Returns
        -------
        str
            an output directory
        """
        if main_directory is None:
            try:
                main_directory = self.experiment.tags[MAIN_DIR_TAG]
            except KeyError as e:
                logger.critical(
                    f"Your experiment has no tag {MAIN_DIR_TAG} pointing to the main directory to use:"
                )
                logger.info("To solve this, add the related tag to the experiment:")
                logger.info(f"mlflow.set_experiment({self.experiment.name})")
                logger.info(
                    f"mlflow.set_experiment_tag({MAIN_DIR_TAG}, YOUR_PROJECT_MAIN_DIRECTORY)"
                )
                raise e
        subfolder = subfolder if subfolder else ""
        experiment_dir = datetime.now().strftime("%Y-%m-%dT%H-%M")
        experiment_dir += ("_" + slugify(suffix)) if suffix else ""
        experiment_dir += ("_" + sha1(metadata)) if metadata else ""
        output_location = osp.join(
            main_directory, self.project_name, self.step_name, subfolder, experiment_dir
        )
        return output_location

