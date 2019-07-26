package ai.libs.mlplan.examples.multiclass.sklearn;

import ai.libs.hasco.model.Component;
import ai.libs.hasco.model.ComponentInstance;
import ai.libs.hasco.model.ComponentUtil;
import ai.libs.hasco.serialization.ComponentLoader;
import ai.libs.jaicore.basic.TimeOut;
import ai.libs.jaicore.ml.core.evaluation.measure.unlabeled.AExternalClusteringValidationMeasure;
import ai.libs.jaicore.ml.core.evaluation.measure.unlabeled.VanDongenMeasure;
import ai.libs.jaicore.ml.evaluation.IInstancesClassifier;
import ai.libs.mlplan.multiclass.wekamlplan.sklearn.SKLearnClusterClassifierFactory;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

public class RandomClusterExample {

	private static final Logger L = LoggerFactory.getLogger(ai.libs.mlplan.examples.multiclass.sklearn.RandomClusterExample.class);

	private static final File DATASET = new File("softwareconfiguration/mlplan/testrsc/winequality_small.arff");

	private static final AExternalClusteringValidationMeasure EXTERNAL_MEASURE = new VanDongenMeasure();

	private static final TimeOut TIMEOUT = new TimeOut(3000, TimeUnit.SECONDS);

	private static final String FS_SEARCH_SPACE_CONFIG = "softwareconfiguration/mlplan/resources/automl/searchmodels/sklearn/sklearn-cluster-mlplan.json";

	private static final String DEF_REQUESTED_HASCO_INTERFACE = "ClusteringAlgorithm";

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

		// currently only selects a suitable component and parametrizes it, no further refinement
		final Collection<Component> allComponents = new ComponentLoader(new File(FS_SEARCH_SPACE_CONFIG)).getComponents();
		final Collection<Component> components = ComponentUtil.getComponentsProvidingInterface(allComponents, DEF_REQUESTED_HASCO_INTERFACE);
		final Random random = new Random();
		//random.setSeed(7353);
		final Component component = components.stream().skip(random.nextInt(components.size())).findFirst().get();
		final ComponentInstance groundComponent = ComponentUtil.randomParameterizationOfComponent(component, random);
		final SKLearnClusterClassifierFactory classifierFactory = new SKLearnClusterClassifierFactory();
		final Classifier classifier = classifierFactory.getComponentInstantiation(groundComponent);
		final IInstancesClassifier instancesClassifier;
		if (classifier instanceof IInstancesClassifier) {
			instancesClassifier = (IInstancesClassifier) classifier;
			classifier.buildClassifier(data);
		} else {
			System.out.println("error");
			return;
		}
		//evaluate results
		final List<Double> actual = Arrays.stream(instancesClassifier.classifyInstances(data)).boxed().collect(Collectors.toList());
		final List<Double> expected = labels.stream().map(x -> x.classValue()).collect(Collectors.toList());

		final double loss = EXTERNAL_MEASURE.calculateExternalMeasure(actual, expected);
		System.out
			.println("Random clustering algorithm has been chosen: " + groundComponent.getComponent().getName() + " with parameters " + groundComponent.getParameterValues() + " with loss: " + loss);
	}
}
