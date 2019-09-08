package ai.libs.jaicore.ml.scikitwrapper;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;

import ai.libs.jaicore.ml.WekaUtil;
import ai.libs.jaicore.ml.openml.OpenMLHelper;
import ai.libs.jaicore.ml.scikitwrapper.ScikitLearnWrapper.ProblemType;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import org.junit.Test;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader.ArffReader;

/**
 * REQUIREMENTS: python 3.6.4 + scikit-learn 0.20.0 need to be installed in order to run these tests.
 *
 * @author Marcel
 */
public class ScikitLearnWrapperTest {

	private static final String MSG_MODELPATH_NOT_NULL = "Model path must not be null.";

	private static final String BASE_TESTRSC_PATH = "testrsc/ml/scikitwrapper/";
	private static final String REGRESSION_ARFF = BASE_TESTRSC_PATH + "0532052678.arff";
	private static final String CLASSIFICATION_ARFF = BASE_TESTRSC_PATH + "dataset_31_credit-g.arff";
	private static final String BAYESNET_TRAIN_ARFF = BASE_TESTRSC_PATH + "Bayesnet_Train.arff";
	private static final String MLP_REGRESSOR_DUMP = BASE_TESTRSC_PATH + "01673183575_MLPRegressor.pcl";
	private static final String CLASSIFIER_DUMP = BASE_TESTRSC_PATH + "0800955787_Pipeline.pcl";
	private static final String CLUSTERING_DUMP = BASE_TESTRSC_PATH + "01868404029_KMeans.pcl";
	private static final String OWN_CLASSIFIER_DUMP = BASE_TESTRSC_PATH + "0532052678.arff";
	private static final String IMPORT_FOLDER = BASE_TESTRSC_PATH + "importfolder_test";

	@Test
	public void buildClassifierRegression() throws Exception {
		final ScikitLearnWrapper slw = new ScikitLearnWrapper("LinearRegression()", "from sklearn.linear_model import LinearRegression");
		final Instances dataset = this.loadARFF(REGRESSION_ARFF);
		slw.setProblemType(ProblemType.REGRESSION);
		slw.buildClassifier(dataset);
		assertNotNull(MSG_MODELPATH_NOT_NULL, slw.getModelPath());
		assertTrue(slw.getModelPath().exists());
	}

	@Test
	public void buildAndPredict() throws Exception {
		final List<String> imports = Arrays.asList("sklearn", "sklearn.ensemble");
		final String constructInstruction = "sklearn.ensemble.RandomForestClassifier(n_estimators=100)";
		final ScikitLearnWrapper slw = new ScikitLearnWrapper(constructInstruction, ScikitLearnWrapper.getImportString(imports), false);
		final Instances dataset = this.loadARFF(CLASSIFICATION_ARFF);
		final List<Instances> stratifiedSplit = WekaUtil.getStratifiedSplit(dataset, 0, .7);

		final long startTrain = System.currentTimeMillis();
		slw.buildClassifier(stratifiedSplit.get(0));
		System.out.println("Build took: " + (System.currentTimeMillis() - startTrain));

		final long startVal = System.currentTimeMillis();
		slw.classifyInstances(stratifiedSplit.get(1));
		System.out.println("Validation took: " + (System.currentTimeMillis() - startVal));
	}

	@Test
	public void buildClassifierRegressionMultitarget() throws Exception {
		final ScikitLearnWrapper slw = new ScikitLearnWrapper("MLPRegressor(activation='logistic')", "from sklearn.neural_network import MLPRegressor");
		final Instances dataset = this.loadARFF(REGRESSION_ARFF);
		slw.setProblemType(ProblemType.REGRESSION);
		final int s = dataset.numAttributes();
		slw.setTargets(s - 1, s - 2, s - 3);
		slw.buildClassifier(dataset);
		assertNotNull(MSG_MODELPATH_NOT_NULL, slw.getModelPath());
		assertTrue(slw.getModelPath().exists());
	}

	@Test
	public void trainAndTestClassifierRegressionMultitarget() throws Exception {
		final ScikitLearnWrapper slw = new ScikitLearnWrapper("MLPRegressor()", "from sklearn.neural_network import MLPRegressor");
		final Instances datasetTrain = this.loadARFF(BAYESNET_TRAIN_ARFF);
		final Instances datasetTest = datasetTrain;
		slw.setProblemType(ProblemType.REGRESSION);
		final int s = datasetTrain.numAttributes();
		final int[] targetColumns = {s - 1, s - 2, s - 3};
		slw.setTargets(targetColumns);
		slw.buildClassifier(datasetTrain);
		final double[] result = slw.classifyInstances(datasetTest);
		assertEquals("Unequal length of predictions and number of test instances", result.length, targetColumns.length * datasetTest.size());
	}

	@Test
	public void testClassifierRegression() throws Exception {
		final ScikitLearnWrapper slw = new ScikitLearnWrapper("MLPRegressor()", "from sklearn.neural_network import MLPRegressor");
		final Instances datasetTest = this.loadARFF(BAYESNET_TRAIN_ARFF);
		slw.setModelPath(new File(MLP_REGRESSOR_DUMP).getAbsoluteFile());
		slw.setProblemType(ProblemType.REGRESSION);
		final double[] result = slw.classifyInstances(datasetTest);
		assertEquals("Unequal length of predictions and number of test instances", result.length, datasetTest.size());
	}

	@Test
	public void trainClassifierCategorical() throws Exception {
		final List<String> imports = Arrays.asList("sklearn", "sklearn.pipeline", "sklearn.decomposition", "sklearn.ensemble");
		final String constructInstruction = "sklearn.pipeline.make_pipeline(sklearn.pipeline.make_union(sklearn.decomposition.PCA(),sklearn.decomposition.FastICA()),sklearn.ensemble.RandomForestClassifier(n_estimators=100))";
		final ScikitLearnWrapper slw = new ScikitLearnWrapper(constructInstruction, ScikitLearnWrapper.getImportString(imports));
		final Instances dataset = this.loadARFF(CLASSIFICATION_ARFF);
		slw.buildClassifier(dataset);
		System.out.println(slw.getModelPath());
		assertNotNull(MSG_MODELPATH_NOT_NULL, slw.getModelPath());
		assertTrue(slw.getModelPath().exists());
	}

	@Test
	public void trainAndTestClassifierCategorical() throws Exception {
		final List<String> imports = Arrays.asList("sklearn", "sklearn.pipeline", "sklearn.decomposition", "sklearn.ensemble");
		final String constructInstruction = "sklearn.pipeline.make_pipeline(sklearn.pipeline.make_union(sklearn.decomposition.PCA(),sklearn.decomposition.FastICA()),sklearn.ensemble.RandomForestClassifier(n_estimators=100))";
		final ScikitLearnWrapper slw = new ScikitLearnWrapper(constructInstruction, ScikitLearnWrapper.getImportString(imports));
		final Instances datasetTrain = this.loadARFF(CLASSIFICATION_ARFF);
		final Instances datasetTest = datasetTrain;
		slw.buildClassifier(datasetTrain);
		final double[] result = slw.classifyInstances(datasetTest);
		assertEquals("Unequal length of predictions and number of test instances", result.length, datasetTest.size());
	}

	@Test
	public void testClassifierCategorical() throws Exception {
		final List<String> imports = Arrays.asList("sklearn", "sklearn.pipeline", "sklearn.decomposition", "sklearn.ensemble");
		final String constructInstruction = "sklearn.pipeline.make_pipeline(sklearn.pipeline.make_union(sklearn.decomposition.PCA(),sklearn.decomposition.FastICA()),sklearn.ensemble.RandomForestClassifier(n_estimators=100))";
		final ScikitLearnWrapper slw = new ScikitLearnWrapper(constructInstruction, ScikitLearnWrapper.getImportString(imports));
		final Instances datasetTest = this.loadARFF(CLASSIFICATION_ARFF);
		slw.setModelPath(new File(CLASSIFIER_DUMP).getAbsoluteFile());
		final double[] result = slw.classifyInstances(datasetTest);
		assertEquals("Unequal length of predictions and number of test instances", result.length, datasetTest.size());
	}

	@Test
	public void trainClassifierClustering() throws Exception {
		final List<String> imports = Arrays.asList("sklearn", "sklearn.pipeline", "sklearn.decomposition", "sklearn.cluster");
		final String constructInstruction = "sklearn.pipeline.make_pipeline(sklearn.pipeline.make_union(sklearn.decomposition.PCA(),sklearn.decomposition.FastICA()),sklearn.cluster.KMeans(n_clusters=3, random_state=0))";
		final ScikitLearnWrapper slw = new ScikitLearnWrapper(constructInstruction, ScikitLearnWrapper.getImportString(imports));
		slw.setProblemType(ScikitLearnWrapper.ProblemType.CLUSTERING);
		final Instances dataset = this.loadARFF(REGRESSION_ARFF);
		slw.buildClassifier(dataset);
		System.out.println(slw.getModelPath());
		assertNotNull(MSG_MODELPATH_NOT_NULL, slw.getModelPath());
		assertTrue(slw.getModelPath().exists());
	}

	@Test
	public void trainAndTestClassifierClustering() throws Exception {
		final List<String> imports = Arrays.asList("sklearn", "sklearn.pipeline", "sklearn.decomposition", "sklearn.cluster");
		final String constructInstruction = "sklearn.pipeline.make_pipeline(sklearn.pipeline.make_union(sklearn.decomposition.PCA(),sklearn.decomposition.FastICA()),sklearn.cluster.KMeans(n_clusters=3, random_state=0))";
		final ScikitLearnWrapper slw = new ScikitLearnWrapper(constructInstruction, ScikitLearnWrapper.getImportString(imports));
		slw.setProblemType(ScikitLearnWrapper.ProblemType.CLUSTERING);
		final Instances datasetTrain = this.loadARFF(REGRESSION_ARFF);
		final Instances datasetTest = datasetTrain;
		slw.buildClassifier(datasetTrain);
		final double[] result = slw.classifyInstances(datasetTest);
		assertEquals("Unequal length of predictions and number of test instances", result.length, datasetTest.size());
	}

	@Test
	public void testClassifierClustering() throws Exception {
		final List<String> imports = Arrays.asList("sklearn", "sklearn.pipeline", "sklearn.decomposition", "sklearn.cluster");
		final String constructInstruction = "sklearn.pipeline.make_pipeline(sklearn.pipeline.make_union(sklearn.decomposition.PCA(),sklearn.decomposition.FastICA()),sklearn.cluster.KMeans(n_clusters=3, random_state=0))";
		final ScikitLearnWrapper slw = new ScikitLearnWrapper(constructInstruction, ScikitLearnWrapper.getImportString(imports));
		slw.setProblemType(ScikitLearnWrapper.ProblemType.CLUSTERING);
		final Instances datasetTest = this.loadARFF(REGRESSION_ARFF);
		slw.setModelPath(new File(CLUSTERING_DUMP).getAbsoluteFile());
		final double[] result = slw.classifyInstances(datasetTest);
		assertEquals("Unequal length of predictions and number of test instances", result.length, datasetTest.size());
	}

	@Test
	public void getRawOutput() throws Exception {
		final ScikitLearnWrapper slw = new ScikitLearnWrapper("MLPRegressor()", "from sklearn.neural_network import MLPRegressor");
		final Instances datasetTrain = this.loadARFF(BAYESNET_TRAIN_ARFF);
		final Instances datasetTest = this.loadARFF(BAYESNET_TRAIN_ARFF);
		slw.setProblemType(ProblemType.REGRESSION);
		final int s = datasetTrain.numAttributes();
		slw.setTargets(s - 1, s - 2, s - 3);
		slw.buildClassifier(datasetTrain);
		slw.classifyInstances(datasetTest);
		assertNotNull(MSG_MODELPATH_NOT_NULL, slw.getModelPath());
		assertTrue(slw.getModelPath().exists());
	}

	@Test
	public void loadOwnClassifierFromFileWithNamespace() throws Exception {
		final File importfolder = new File(IMPORT_FOLDER);
		final String importStatement = ScikitLearnWrapper.createImportStatementFromImportFolder(importfolder, true);
		final ScikitLearnWrapper slw = new ScikitLearnWrapper("test_module_1.My_MLPRegressor()", importStatement);
		final Instances dataset = this.loadARFF(REGRESSION_ARFF);
		slw.setProblemType(ProblemType.REGRESSION);
		final int s = dataset.numAttributes();
		slw.setTargets(s - 1, s - 2, s - 3);
		slw.buildClassifier(dataset);
		assertNotNull(MSG_MODELPATH_NOT_NULL, slw.getModelPath());
		assertTrue(slw.getModelPath().exists());
	}

	@Test
	public void loadOwnClassifierFromFileWithoutNamespace() throws Exception {
		final File importfolder = new File(IMPORT_FOLDER);
		final String importStatement = ScikitLearnWrapper.createImportStatementFromImportFolder(importfolder, false);
		final ScikitLearnWrapper slw = new ScikitLearnWrapper("My_MLPRegressor()", importStatement);
		final Instances dataset = this.loadARFF(OWN_CLASSIFIER_DUMP);
		slw.setProblemType(ProblemType.REGRESSION);
		final int s = dataset.numAttributes();
		slw.setTargets(s - 1, s - 2, s - 3);
		slw.buildClassifier(dataset);
		assertNotNull(MSG_MODELPATH_NOT_NULL, slw.getModelPath());
		assertTrue(slw.getModelPath().exists());
	}

	@Test
	public void invalidConstructorNoConstructionCall() throws IOException {
		boolean errorTriggeredFlag = false;
		try {
			new ScikitLearnWrapper(null, "");
		} catch (final AssertionError e) {
			errorTriggeredFlag = true;
		}
		assertTrue(errorTriggeredFlag);
	}

	@Test
	public void invalidConstructorEmptyConstructionCall() throws IOException {
		boolean errorTriggeredFlag = false;
		try {
			new ScikitLearnWrapper("", "");
		} catch (final AssertionError e) {
			errorTriggeredFlag = true;
		}
		assertTrue(errorTriggeredFlag);
	}

	private Instances loadARFF(final String arffPath) throws IOException {
		final BufferedReader reader = new BufferedReader(new FileReader(arffPath));
		final ArffReader arff = new ArffReader(reader);
		final Instances data = arff.getData();
		data.setClassIndex(data.numAttributes() - 1);
		return data;
	}

	@Test
	public void kmeansTest() throws Exception {
		final List<String> imports = Arrays.asList("sklearn.cluster");
		final String constructInstruction = "sklearn.cluster.KMeans(n_clusters=3, random_state=0)";
		final ScikitLearnWrapper slw = new ScikitLearnWrapper(constructInstruction, ScikitLearnWrapper.getImportString(imports));
		slw.setProblemType(ScikitLearnWrapper.ProblemType.CLUSTERING);

		final Instances labeledData = OpenMLHelper.getInstancesById(61);//this.loadARFF(REGRESSION_ARFF);
		//strip labels
		final ArrayList<Attribute> atts = Collections.list(labeledData.enumerateAttributes());
		final Instances unlabeledData = new Instances("data", atts, labeledData.size());

		labeledData.setClassIndex(labeledData.numAttributes() - 1);
		for (final Instance instance :
			labeledData) {
			final double[] values = new double[atts.size()];
			for (int i = 0; i < values.length; i++) {

				values[i] = instance.value(i);

			}
			final Instance newInstance = new DenseInstance(1, values);
			unlabeledData.add(newInstance);
			newInstance.setDataset(unlabeledData);
		}
		unlabeledData.setClassIndex(-1);//ignored, but needed for mlplan

		slw.buildClassifier(unlabeledData);
		final double[] result = slw.classifyInstances(unlabeledData);
		final HashMap<Double, Integer> classes = new HashMap<>();
		for (final double label : result
		) {
			if (classes.containsKey(label)) {
				classes.put(label, classes.get(label) + 1);
			} else {
				classes.put(label, 1);
			}
		}
		assertEquals("Unequal length of predictions and number of test instances", 3, classes.size());
	}
}
