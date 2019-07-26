package ai.libs.jaicore.ml.core.evaluation.measure.unlabeled;

import java.util.List;
import java.util.function.DoubleBinaryOperator;


/*
based on halkich, dataset oriented approach
 */
public class DensityBasedMeasure extends AInternalClusteringValidationMeasure {


	@Override
	public Double calculateMeasure(final List<List<double[]>> clusters) {
		return 0d;
		//sreturn calculateInterClusterDensity(clusters) + calculateIntraClusterVariance(clusters);
	}

	private Double calculateStandardDeviation(final List<List<double[]>> clusters) {
		return Math.sqrt(clusters.stream().mapToDouble(x -> calculateSigma(x)).reduce(new DoubleBinaryOperator() {
			@Override
			public double applyAsDouble(final double a, final double b) {
				return a * b;
			}
		}).getAsDouble()) / clusters.size();
	}

	private Double calculateSigma(final List<double[]> cluster) {
		return 0d;
	}

	private Double calculateInterClusterDensity(final List<List<double[]>> clusters) {
		final double c = clusters.size();
		double dens_bw = 1 / (c * (c - 1));
		for (int i = 0; i < c; i++) {
			for (int j = 0; j < c; j++) {
				if (i != j) {
					final double[] vi = this.centroids.get(i);
					final double[] vj = this.centroids.get(j);
					final double[] uij = new double[vi.length];
					for (int k = 0; k < vi.length; k++) {
						uij[k] = (vi[k] + vj[k]) / 2;
					}
					dens_bw *= calculateDensity(uij) / Math.max(calculateDensity(vi), calculateDensity(vj));
				}
			}
		}
		return dens_bw;
	}

	private double calculateDensity(final double[] point) {
		return 0d;
	}

	private Double calculateIntraClusterVariance(final List<List<double[]>> clusters) {
		//total cluster variance / data set variance
		return 0d;
	}

}
