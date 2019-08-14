package ai.libs.mlplan;

import ai.libs.jaicore.ml.core.evaluation.measure.unlabeled.AExternalClusteringValidationMeasure;
import ai.libs.jaicore.ml.core.evaluation.measure.unlabeled.IInternalClusteringValidationMeasure;
import ai.libs.jaicore.ml.core.evaluation.measure.unlabeled.VanDongenMeasure;
import ai.libs.jaicore.ml.evaluation.IInstancesClassifier;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.stream.Collectors;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Add;

public class ClusteringEvaluation {

	/**
	 * Evaluates a clustering using an external measure.
	 */
	private static final Logger L = LoggerFactory.getLogger(ClusteringEvaluation.class);
	private static final AExternalClusteringValidationMeasure EXTERNAL_MEASURE = new VanDongenMeasure();

	public static ClusteringResult evaluateModel(final Classifier classifier, final Instances labeledData, final IInternalClusteringValidationMeasure internalMeasure) {

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
		//unlabeledData.setClassIndex(-1);//ignored, but needed for mlplan

		// train classifier on unlabeled data
		try {
			classifier.buildClassifier(unlabeledData);
		} catch (final Exception e) {
			L.error(e.getMessage());
		}
		final IInstancesClassifier instancesClassifier;
		if (classifier instanceof IInstancesClassifier) {
			instancesClassifier = (IInstancesClassifier) classifier;
		} else {
			throw new IllegalArgumentException("The given Classifier does not implement the IInstancesClassifier interface needed for clustering.");
		}

		final ClusteringResult cr = new ClusteringResult();
		cr.setInternalEvaluationResult(Double.MAX_VALUE);

		// evaluate external cluster validation measure
		final List<Double> actual;
		final double[] results;
		try {
			results = instancesClassifier.classifyInstances(unlabeledData);
		} catch (final Exception e) {
			L.error(e.getMessage());
			return cr;
		}
		actual = Arrays.stream(results).boxed().collect(Collectors.toList());

		final List<Double> expected = labeledData.stream().map(x -> x.classValue()).collect(Collectors.toList());

		final double loss = EXTERNAL_MEASURE.calculateExternalMeasure(actual, expected);

		cr.setExternalEvaluationResult(loss);
		cr.setResultValid(false);

		//calculate internal cluster validation measure
		Instances classifiedData = new Instances(unlabeledData);
		final Add add = new Add();
		add.setAttributeName("label");
		add.setAttributeType(new SelectedTag("NUM", Add.TAGS_TYPE));
		add.setAttributeIndex("last");
		try {
			add.setInputFormat(classifiedData);
		} catch (final Exception e) {
			e.printStackTrace();
		}

		try {
			classifiedData = Filter.useFilter(classifiedData, add);
		} catch (final Exception e) {
			e.printStackTrace();
		}
		classifiedData.setClassIndex(classifiedData.numAttributes() - 1);

		final HashMap<Double, Integer> classes = new HashMap<>();

		for (int i = 0; i < classifiedData.size(); i++) {
			classifiedData.instance(i).setValue(classifiedData.classIndex(), results[i]);
			if (classes.containsKey(results[i])) {
				classes.put(results[i], classes.get(results[i]) + 1);
			} else {
				classes.put(results[i], 1);
			}
		}

		cr.setResultValid(classes.size() > 1);
		cr.setN_clusters(classes.size());

		try {
			cr.setInternalEvaluationResult(internalMeasure.calculateMeasure(classifiedData, null));
		} catch (final Exception e) {
			e.printStackTrace();
		}

		return cr;
	}

	static class ClusteringResult {

		double externalEvaluationResult;
		double internalEvaluationResult;
		boolean resultValid;
		int n_clusters;

		public double getExternalEvaluationResult() {
			return this.externalEvaluationResult;
		}

		void setExternalEvaluationResult(final double externalEvaluationResult) {
			this.externalEvaluationResult = externalEvaluationResult;
		}

		double getInternalEvaluationResult() {
			return this.internalEvaluationResult;
		}

		void setInternalEvaluationResult(final double internalEvaluationResult) {
			this.internalEvaluationResult = internalEvaluationResult;
		}

		boolean isResultValid() {
			return this.resultValid;
		}

		void setResultValid(final boolean resultValid) {
			this.resultValid = resultValid;
		}

		public int getN_clusters() {
			return this.n_clusters;
		}

		void setN_clusters(final int n_clusters) {
			this.n_clusters = n_clusters;
		}
	}
}
