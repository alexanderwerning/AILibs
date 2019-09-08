package ai.libs.jaicore.ml.core.evaluation.measure.unlabeled;

import java.io.FileReader;
import java.io.IOException;
import java.util.Random;
import org.junit.Test;
import weka.core.Instance;
import weka.core.Instances;

public class InternalMeasureTest {

	private final String irisFilename = "testrsc/ml/core/evaluation/measure/unlabeled/dataset_61_iris.arff";

	@Test
	public void test() {
		final IInternalClusteringValidationMeasure[] measures = new IInternalClusteringValidationMeasure[]{new SquaredErrorMeasure(), new SilhouetteMeasure(), new DunnMeasure(),
			new DaviesBouldinMeasure(),
			new DensityBasedMeasure()};
		for (final IInternalClusteringValidationMeasure measure : measures) {

			try {
				final Instances instances = new Instances(new FileReader(this.irisFilename));
				instances.setClassIndex(instances.numAttributes() - 1);
				final double lossBestResult = measure.calculateMeasure(instances, null);
				final Random random = new Random(2);
				for (final Instance instance : instances) {
					instance.setValue(instances.classIndex(), random.nextInt(3));
				}
				final double lossWorstResult = measure.calculateMeasure(instances, null);
				System.out.println(lossBestResult + " " + lossWorstResult);
				//assertTrue(lossBestResult < lossWorstResult);

			} catch (final IOException e) {
				e.printStackTrace();
			}
		}
	}
}
