package ai.libs.jaicore.ml.evaluation.evaluators.weka;


import ai.libs.jaicore.basic.ILoggingCustomizable;
import ai.libs.jaicore.basic.algorithm.exceptions.ObjectEvaluationFailedException;
import ai.libs.jaicore.ml.core.evaluation.measure.IMeasure;
import ai.libs.jaicore.ml.core.evaluation.measure.unlabeled.IInternalClusteringMeasure;
import java.util.ArrayList;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

public class ClusteringValidationEvaluator implements IClassifierEvaluator, ILoggingCustomizable {
	private Logger logger = LoggerFactory.getLogger(MonteCarloCrossValidationEvaluator.class);
	private final Instances data;
	private IMeasure<Instances,Double> evaluator;

	public ClusteringValidationEvaluator(final Instances data) {
		if (data == null) {
			throw new IllegalArgumentException("Cannot work with NULL data");
		}
		this.data = data;
	}

	@Override
	public Double evaluate(final Classifier object) throws InterruptedException, ObjectEvaluationFailedException {

		try {
			object.buildClassifier(data);
		} catch (ObjectEvaluationFailedException | InterruptedException e) {
			throw e;
		} catch (Exception e) {
			throw new ObjectEvaluationFailedException("Could not train classifier");
		}
		if(data.classIndex() != -1){
			// label attribute is included, external validation possible
			Instances actual = new Instances(data);
			try {
				for (int i = 0; i < actual.size(); i++) {
					actual.get(i).setValue(actual.classIndex(), object.classifyInstance(actual.get(i)));
				}
			} catch (Exception e) {
				e.printStackTrace();
			}
			return evaluator.calculateMeasure(actual, data);
		}else{
			// no label, internal validation
			if(!(this.evaluator instanceof IInternalClusteringMeasure)){
				throw new IllegalArgumentException("Received data without labels for an external cluster validation measure.");
			}

			ArrayList<Attribute> atts = new ArrayList<>();
			atts.add(new Attribute("label"));
			Instances labels = new Instances("labels", atts, 0);

			try {
				for (int i = 0; i < data.size(); i++) {
					double[] values = new double[1];
					values[0] = object.classifyInstance(data.get(i));
					Instance instance = new DenseInstance(1,values);
					labels.add(instance);
					instance.setDataset(labels);
				}
			} catch (Exception e) {
				e.printStackTrace();
			}
			return evaluator.calculateMeasure(labels, data);
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
