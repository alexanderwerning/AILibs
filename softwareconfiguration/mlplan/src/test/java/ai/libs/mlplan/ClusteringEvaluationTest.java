package ai.libs.mlplan;

import ai.libs.jaicore.ml.core.evaluation.measure.unlabeled.SilhouetteMeasure;
import ai.libs.jaicore.ml.openml.OpenMLHelper;
import ai.libs.jaicore.ml.scikitwrapper.ScikitLearnWrapper;
import ai.libs.mlplan.unlabeled.ClusteringEvaluation;
import ai.libs.mlplan.unlabeled.ClusteringEvaluation.ClusteringResult;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import org.apache.commons.lang3.StringUtils;
import org.junit.Test;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

public class ClusteringEvaluationTest {

	@Test
	public void test1() throws Exception {
		final List<String> imports = Arrays.asList("sklearn.cluster");
		final String constructInstruction = "sklearn.cluster.KMeans(n_clusters=3, random_state=0)";
		final ScikitLearnWrapper slw = new ScikitLearnWrapper(constructInstruction, "import " + StringUtils.join(imports, "\nimport "));
		slw.setProblemType(ScikitLearnWrapper.ProblemType.CLUSTERING);
		final Instances labeledData = OpenMLHelper.getInstancesById(61);
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
		final ClusteringResult cr = ClusteringEvaluation.evaluateModel(slw, labeledData, new SilhouetteMeasure());

		//assertEquals("Unequal length of predictions and number of test instances", 3, classes.size());
	}
}
