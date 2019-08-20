package ai.libs.jaicore.ml.evaluation.evaluators.weka;


import ai.libs.jaicore.basic.ILoggingCustomizable;
import ai.libs.jaicore.basic.algorithm.exceptions.ObjectEvaluationFailedException;
import ai.libs.jaicore.ml.core.evaluation.measure.IMeasure;
import ai.libs.jaicore.ml.core.evaluation.measure.unlabeled.IInternalClusteringValidationMeasure;
import ai.libs.jaicore.ml.evaluation.IInstancesClassifier;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import weka.classifiers.Classifier;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Add;

public class ClusteringValidationEvaluator implements IClassifierEvaluator, ILoggingCustomizable {

	private Logger logger = LoggerFactory.getLogger(MonteCarloCrossValidationEvaluator.class);
	private final Instances data;
	private final IMeasure<Instances, Double> evaluator;

	public ClusteringValidationEvaluator(final IMeasure<Instances, Double> evaluator, final Instances data) {
		if (data == null) {
			throw new IllegalArgumentException("Cannot work with NULL data");
		}
		this.data = data;
		this.evaluator = evaluator;
	}

	@Override
	public Double evaluate(final Classifier object) throws InterruptedException, ObjectEvaluationFailedException {
		if (!(object instanceof IInstancesClassifier)) {
			throw new IllegalArgumentException("The classifier needs to implement IInstancesClassifier for the evaluation");
		}
		final IInstancesClassifier classifier = (IInstancesClassifier) object;
		try {
			object.buildClassifier(this.data);
		} catch (final ObjectEvaluationFailedException | InterruptedException e) {
			throw e;
		} catch (final Exception e) {
			throw new ObjectEvaluationFailedException("Could not train classifier");
		}

		if (this.data.classIndex() != -1) {
			// label attribute is included, external validation possible

			final Instances actual = new Instances(this.data);
			try {
				final double[] results = classifier.classifyInstances(this.data);
				for (int i = 0; i < actual.size(); i++) {
					actual.get(i).setValue(actual.classIndex(), results[i]);
				}
			} catch (final Exception e) {
				e.printStackTrace();
			}
			return this.evaluator.calculateMeasure(actual, this.data);
		} else {
			// no label, internal validation
			if (!(this.evaluator instanceof IInternalClusteringValidationMeasure)) {
				throw new IllegalArgumentException("Received data without labels for an external cluster validation measure.");
			}
/*
			final ArrayList<Attribute> atts = new ArrayList<>();
			atts.add(new Attribute("label"));
			final Instances labels = new Instances("labels", atts, 0);
			labels.setClassIndex(0);*/
			final Add add = new Add();
			add.setAttributeName("label");
			add.setAttributeType(new SelectedTag("NUM", Add.TAGS_TYPE));
			add.setAttributeIndex("last");
			try {
				add.setInputFormat(this.data);
			} catch (final Exception e) {
				e.printStackTrace();
			}

			Instances labeled = null;
			try {
				labeled = Filter.useFilter(this.data, add);
			} catch (final Exception e) {
				e.printStackTrace();
			}
			labeled.setClassIndex(labeled.numAttributes() - 1);

			try {
				final double[] results = classifier.classifyInstances(this.data);
				for (int i = 0; i < this.data.size(); i++) {
					labeled.instance(i).setValue(labeled.classIndex(), results[i]);
				}
			} catch (final Exception e) {
				e.printStackTrace();
			}
			return this.evaluator.calculateMeasure(labeled, null);
		}

	}

	@Override
	public String getLoggerName() {
		return this.logger.getName();
	}

	@Override
	public void setLoggerName(final String name) {
		this.logger.info("Switching logger of {} from {} to {}", this, this.logger.getName(), name);
		this.logger = LoggerFactory.getLogger(name);
		this.logger.info("Switched logger of {} to {}", this, name);
	}
}
