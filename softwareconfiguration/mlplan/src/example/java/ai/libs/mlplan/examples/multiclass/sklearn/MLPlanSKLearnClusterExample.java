package ai.libs.mlplan.examples.multiclass.sklearn;

import ai.libs.jaicore.basic.TimeOut;
import ai.libs.jaicore.ml.core.evaluation.measure.unlabeled.AExternalClusteringValidationMeasure;
import ai.libs.jaicore.ml.core.evaluation.measure.unlabeled.DaviesBouldinMeasure;
import ai.libs.jaicore.ml.core.evaluation.measure.unlabeled.VanDongenMeasure;
import ai.libs.mlplan.core.AbstractMLPlanBuilder;
import ai.libs.mlplan.core.MLPlanSKLearnClusterBuilder;
import ai.libs.mlplan.multiclass.MLPlanClassifierConfig;
import ai.libs.mlplan.multiclass.wekamlplan.sklearn.SKLearnClusterMLPlanWekaClassifier;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

public class MLPlanSKLearnClusterExample {

	private static final Logger L = LoggerFactory.getLogger(MLPlanSKLearnClusterExample.class);

	private static final File DATASET = new File("softwareconfiguration/mlplan/testrsc/winequality_small.arff");
	private static final AExternalClusteringValidationMeasure EXTERNAL_MEASURE = new VanDongenMeasure();

	private static final TimeOut TIMEOUT = new TimeOut(3000, TimeUnit.SECONDS);

	public static void main(final String[] args) throws Exception {
		//get labeled data
		final Instances labels = new Instances(new FileReader(DATASET));
		//strip labels
		final ArrayList<Attribute> atts = Collections.list(labels.enumerateAttributes());
		atts.remove(atts.size() - 1);
		final Instances data = new Instances("data", atts, labels.size());

		labels.setClassIndex(labels.numAttributes() - 1);
		for (final Instance instance :
			labels) {
			final double[] values = new double[labels.numAttributes() - 1];
			for (int i = 0; i < values.length; i++) {
				values[i] = instance.value(i);
			}
			final Instance newInstance = new DenseInstance(1, values);
			data.add(newInstance);
			newInstance.setDataset(data);
		}
		data.setClassIndex(-1);//ignored, but needed for mlplan

		final AbstractMLPlanBuilder builder = AbstractMLPlanBuilder.forSKLearnCluster();
		builder.withTimeOut(TIMEOUT);
		builder.withNodeEvaluationTimeOut(new TimeOut(900, TimeUnit.SECONDS));
		builder.withCandidateEvaluationTimeOut(new TimeOut(60, TimeUnit.SECONDS));
		((MLPlanSKLearnClusterBuilder) builder).withValidationMeasure(new DaviesBouldinMeasure());

		final SKLearnClusterMLPlanWekaClassifier mlplan = new SKLearnClusterMLPlanWekaClassifier(builder);
		mlplan.setLoggerName("sklmlplanc");
		mlplan.setVisualizationEnabled(true);
		mlplan.getMLPlanConfig().setProperty(MLPlanClassifierConfig.SELECTION_PORTION, "0");//set selection split to 0
		mlplan.buildClassifier(data);
		//evaluate results
		final List<Double> actual = Arrays.stream(mlplan.classifyInstances(data)).mapToObj(x -> x).collect(Collectors.toList());
		final List<Double> expected = labels.stream().map(x -> x.classValue()).collect(Collectors.toList());

		final double loss = EXTERNAL_MEASURE.calculateExternalMeasure(actual, expected);
		System.out.println("ML-Plan classifier has been chosen for dataset " + DATASET.getAbsolutePath() + " and framework SK-Learn. The measured test loss of the selected classifier is " + loss);
	}

}
