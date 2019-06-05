package de.upb.crc901.mlplan.core;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import de.upb.crc901.mlplan.multiclass.wekamlplan.IClassifierFactory;
import de.upb.crc901.mlplan.multiclass.wekamlplan.sklearn.SKLearnClassifierFactory;
import jaicore.basic.FileUtil;
import jaicore.basic.SystemRequirementsNotMetException;
import jaicore.basic.sets.SetUtil;
import jaicore.ml.evaluation.evaluators.weka.factory.MonteCarloCrossValidationEvaluatorFactory;
import jaicore.ml.evaluation.evaluators.weka.splitevaluation.SimpleSLCSplitBasedClassifierEvaluator;
import jaicore.ml.weka.dataset.splitter.IDatasetSplitter;
import jaicore.ml.weka.dataset.splitter.MulticlassClassStratifiedSplitter;

public class MLPlanSKLearnClusterBuilder extends MLPlanSKLearnBuilder {

	private Logger logger = LoggerFactory.getLogger(MLPlanSKLearnClusterBuilder.class);

	private static final String PYTHON_REQUIRED_VERSION = "Python 3.7.*";
	private static final String[] PYTHON_REQUIRED_MODULES = { "numpy", "json", "pickle", "os", "sys", "warnings", "scipy.io.arff", "sklearn" };

	private static final String COMMAND_PYTHON = "python";
	private static final String[] COMMAND_PYTHON_VERSION = { COMMAND_PYTHON, "--version" };
	private static final String[] COMMAND_PYTHON_EXEC = { COMMAND_PYTHON, "-c" };
	private static final String PYTHON_MODULE_NOT_FOUND_ERROR_MSG = "ModuleNotFoundError";

	/* DEFAULT VALUES FOR THE SCIKIT-LEARN SETTING */
	private static final String RES_SKLEARN_SEARCHSPACE_CONFIG = "automl/searchmodels/sklearn/sklearn-cluster-mlplan.json";
	private static final String RES_SKLEARN_UL_SEARCHSPACE_CONFIG = RES_SKLEARN_SEARCHSPACE_CONFIG;//"automl/searchmodels/sklearn/ml-plan-ul.json";
	private static final String FS_SEARCH_SPACE_CONFIG = "conf/mlplan-sklearn.json";

	private static final String RES_SKLEARN_PREFERRED_COMPONENTS = "mlplan/sklearn-preferenceList.txt";
	private static final String FS_SKLEARN_PREFERRED_COMPONENTS = "conf/sklearn-preferenceList.txt";

	private static final String DEF_REQUESTED_HASCO_INTERFACE = "ClusteringAlgorithm";
	private static final IDatasetSplitter DEF_SELECTION_HOLDOUT_SPLITTER = new MulticlassClassStratifiedSplitter();
	private static final IClassifierFactory DEF_CLASSIFIER_FACTORY = new SKLearnClassifierFactory();
	private static final File DEF_SEARCH_SPACE_CONFIG = FileUtil.getExistingFileWithHighestPriority(RES_SKLEARN_SEARCHSPACE_CONFIG, FS_SEARCH_SPACE_CONFIG);
	private static final File DEF_PREFERRED_COMPONENTS = FileUtil.getExistingFileWithHighestPriority(RES_SKLEARN_PREFERRED_COMPONENTS, FS_SKLEARN_PREFERRED_COMPONENTS);
	private static final MonteCarloCrossValidationEvaluatorFactory DEF_SEARCH_PHASE_EVALUATOR = new MonteCarloCrossValidationEvaluatorFactory().withNumMCIterations(SEARCH_NUM_MC_ITERATIONS).withTrainFoldSize(SEARCH_TRAIN_FOLD_SIZE)
			.withSplitBasedEvaluator(new SimpleSLCSplitBasedClassifierEvaluator(LOSS_FUNCTION)).withDatasetSplitter(new MulticlassClassStratifiedSplitter());
	private static final MonteCarloCrossValidationEvaluatorFactory DEF_SELECTION_PHASE_EVALUATOR = new MonteCarloCrossValidationEvaluatorFactory().withNumMCIterations(SELECTION_NUM_MC_ITERATIONS).withTrainFoldSize(SELECTION_TRAIN_FOLD_SIZE)
			.withSplitBasedEvaluator(new SimpleSLCSplitBasedClassifierEvaluator(LOSS_FUNCTION)).withDatasetSplitter(new MulticlassClassStratifiedSplitter());

	/**
	 * Creates a new ML-Plan Builder for scikit-learn.
	 * @throws IOException Thrown if configuration files cannot be read.
	 */
	public MLPlanSKLearnClusterBuilder() throws IOException {
		this(false);
	}

	/**
	 * Creates a new ML-Plan Builder for scikit-learn.
	 *
	 * @param skipSetupCheck Flag whether to skip the system's setup check, which examines whether the operating system has python installed in the required version and all the required python modules are installed.
	 * @throws IOException Thrown if configuration files cannot be read.
	 */
	public MLPlanSKLearnClusterBuilder(final boolean skipSetupCheck) throws IOException {
		super();
		if (!skipSetupCheck) {
			this.checkPythonSetup();
		}
		this.withSearchSpaceConfigFile(DEF_SEARCH_SPACE_CONFIG);
		this.withPreferredComponentsFile(DEF_PREFERRED_COMPONENTS);
		this.withRequestedInterface(DEF_REQUESTED_HASCO_INTERFACE);
		this.withClassifierFactory(DEF_CLASSIFIER_FACTORY);
		this.withDatasetSplitterForSearchSelectionSplit(DEF_SELECTION_HOLDOUT_SPLITTER);
		this.withSearchPhaseEvaluatorFactory(DEF_SEARCH_PHASE_EVALUATOR);
		this.withSelectionPhaseEvaluatorFactory(DEF_SELECTION_PHASE_EVALUATOR);
		this.setPerformanceMeasureName(LOSS_FUNCTION.getClass().getSimpleName());
	}
 
	/**
	 * Configures ML-Plan to use the search space with unlimited length preprocessing pipelines.
	 * @return The builder object.
	 * @throws IOException Thrown if the search space configuration file cannot be read.
	 */
	public MLPlanSKLearnClusterBuilder withUnlimitedLengthPipelineSearchSpace() throws IOException {
		return (MLPlanSKLearnClusterBuilder) this.withSearchSpaceConfigFile(FileUtil.getExistingFileWithHighestPriority(RES_SKLEARN_UL_SEARCHSPACE_CONFIG, FS_SEARCH_SPACE_CONFIG));
	}

	@Override
	protected IDatasetSplitter getDefaultDatasetSplitter() {
		return new MulticlassClassStratifiedSplitter();
	}
}
