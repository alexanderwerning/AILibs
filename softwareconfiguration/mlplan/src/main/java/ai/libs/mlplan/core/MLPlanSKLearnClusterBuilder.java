package ai.libs.mlplan.core;

import ai.libs.jaicore.ml.evaluation.evaluators.weka.factory.ClusteringValidationEvaluationFactory;

import java.io.File;
import java.io.IOException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ai.libs.jaicore.basic.FileUtil;

import ai.libs.jaicore.ml.weka.dataset.splitter.IDatasetSplitter;
import ai.libs.jaicore.ml.weka.dataset.splitter.MulticlassClassStratifiedSplitter;
import ai.libs.mlplan.multiclass.wekamlplan.IClassifierFactory;
import ai.libs.mlplan.multiclass.wekamlplan.sklearn.SKLearnClassifierFactory;

public class MLPlanSKLearnClusterBuilder extends MLPlanSKLearnBuilder {

	private Logger logger = LoggerFactory.getLogger(MLPlanSKLearnClusterBuilder.class);

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
	private static final ClusteringValidationEvaluationFactory DEF_SEARCH_PHASE_EVALUATOR = new ClusteringValidationEvaluationFactory();

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
