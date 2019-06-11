package de.upb.crc901.mlplan.examples.multiclass.sklearn;

import java.io.File;
import java.io.FileReader;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import de.upb.crc901.mlplan.core.AbstractMLPlanBuilder;
import de.upb.crc901.mlplan.multiclass.wekamlplan.sklearn.SKLearnClusterMLPlanWekaClassifier;
import de.upb.crc901.mlplan.multiclass.wekamlplan.sklearn.SKLearnMLPlanWekaClassifier;
import jaicore.basic.TimeOut;
import jaicore.ml.WekaUtil;
import jaicore.ml.core.evaluation.measure.singlelabel.ZeroOneLoss;
import weka.core.Instances;

public class MLPlanSKLearnClusterExample {

	private static final Logger L = LoggerFactory.getLogger(MLPlanSKLearnClusterExample.class);

	private static final File DATASET = new File("testrsc/car.arff");
	//TODO change to external validation measure
	private static final ZeroOneLoss LOSS_MEASURE = new ZeroOneLoss();

	private static final TimeOut TIMEOUT = new TimeOut(300, TimeUnit.SECONDS);

	public static void main(final String[] args) throws Exception {
		Instances data = new Instances(new FileReader(DATASET));
		Instances labels = new Instances(new FileReader(DATASET));
		labels.setClassIndex(labels.numAttributes() - 1);
		data.remove(data.numAttributes() - 1);

		AbstractMLPlanBuilder builder = AbstractMLPlanBuilder.forSKLearnCluster();
		builder.withTimeOut(TIMEOUT);
		builder.withNodeEvaluationTimeOut(new TimeOut(90, TimeUnit.SECONDS));
		builder.withCandidateEvaluationTimeOut(new TimeOut(30, TimeUnit.SECONDS));

		SKLearnClusterMLPlanWekaClassifier mlplan = new SKLearnClusterMLPlanWekaClassifier(builder);
		mlplan.setLoggerName("sklmlplanc");
		mlplan.setVisualizationEnabled(true);
		mlplan.buildClassifier(data);

		List<Double> actual = Arrays.stream(mlplan.classifyInstances(data)).mapToObj(x -> x).collect(Collectors.toList());
		List<Double> expected = labels.stream().map(x -> x.classValue()).collect(Collectors.toList());
		//TODO is avgmeasure still correct?
		double loss = LOSS_MEASURE.calculateAvgMeasure(actual, expected);
		L.info("ML-Plan classifier has been chosen for dataset {} and framework SK-Learn. The measured test loss of the selected classifier is {}", DATASET.getAbsolutePath(), loss);
	}

}
