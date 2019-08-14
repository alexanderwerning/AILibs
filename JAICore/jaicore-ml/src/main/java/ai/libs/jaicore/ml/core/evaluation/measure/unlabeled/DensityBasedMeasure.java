package ai.libs.jaicore.ml.core.evaluation.measure.unlabeled;

import static ai.libs.jaicore.ml.tsc.util.MathUtil.singleSquaredEuclideanDistance;

import java.util.ArrayList;
import java.util.List;
import java.util.function.DoubleBinaryOperator;


/*
based on halkich, dataset oriented approach
 */
public class DensityBasedMeasure extends AInternalClusteringValidationMeasure {

	private double stddev;

	@Override
	public Double calculateMeasure(final List<List<double[]>> clusters) {
		this.stddev = calculateStandardDeviation(clusters);
		final double interClusterDensity = calculateInterClusterDensity(clusters);
		final double intraClusterVariance = calculateIntraClusterVariance(clusters);
		return interClusterDensity + intraClusterVariance;
	}

	private Double calculateStandardDeviation(final List<List<double[]>> clusters) {
		return Math.sqrt(clusters.stream().mapToDouble(x -> calculateSigma(x)).reduce(new DoubleBinaryOperator() {
			@Override
			public double applyAsDouble(final double a, final double b) {
				return a + b;
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
			for (int j = 0; j < i; j++) {
				final double[] vi = this.centroids.get(i);
				final double[] vj = this.centroids.get(j);
				final double[] uij = new double[vi.length];
				for (int k = 0; k < vi.length; k++) {
					uij[k] = (vi[k] + vj[k]) / 2;
				}
				final List<double[]> union = new ArrayList<>();
				union.addAll(clusters.get(i));
				union.addAll(clusters.get(j));
				final double densityUnion = calculateDensity(uij, union);
				final double densityI = calculateDensity(vi, clusters.get(i));
				final double densityJ = calculateDensity(vj, clusters.get(j));
				final double max = Math.max(densityI, densityJ);
				if (max != 0) {
					dens_bw += densityUnion / max;
				}
			}
		}
		return dens_bw;
	}

	private double calculateDensity(final double[] point, final List<double[]> neighbourhood) {
		double sum = 0;
		for (int i = 0; i < neighbourhood.size(); i++) {
			if (Math.sqrt(singleSquaredEuclideanDistance(point, neighbourhood.get(i))) > this.stddev) {
				sum++;
			}
		}
		return sum;
	}

	private Double calculateIntraClusterVariance(final List<List<double[]>> clusters) {
		final int dimensions = clusters.get(0).get(0).length;
		final double[] datasetCentroid = new double[dimensions];
		int instances = 0;
		for (int j = 0; j < clusters.size(); j++) {
			for (int i = 0; i < datasetCentroid.length; i++) {
				datasetCentroid[i] += clusters.get(j).size() * this.centroids.get(j)[i];

			}
			instances += clusters.size();
		}
		for (int i = 0; i < dimensions; i++) {
			datasetCentroid[i] /= instances;
		}

		double sum = 0;
		final double[] zero = new double[dimensions];
		final double[] totalVariance = new double[dimensions];
		for (int i = 0; i < clusters.size(); i++) {
			final List<double[]> cluster = clusters.get(i);
			final double[] variance = new double[dimensions];
			for (int j = 0; j < cluster.size(); j++) {
				for (int k = 0; k < dimensions; k++) {
					variance[k] += Math.pow(this.centroids.get(i)[k] - cluster.get(j)[k], 2);
					totalVariance[k] += Math.pow(datasetCentroid[k] - cluster.get(j)[k], 2);
				}
			}

			sum += singleSquaredEuclideanDistance(variance, zero);
		}
		sum /= clusters.size();
		final double datasetVariance = singleSquaredEuclideanDistance(totalVariance, zero) / instances;

		return sum / datasetVariance;
	}

}
