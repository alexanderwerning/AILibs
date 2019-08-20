package ai.libs.jaicore.ml.core.evaluation.measure.unlabeled;

import static ai.libs.jaicore.ml.tsc.util.MathUtil.singleSquaredEuclideanDistance;

import java.util.ArrayList;
import java.util.List;


/*
based on halkich, dataset oriented approach
 */
public class DensityBasedMeasure extends AInternalClusteringValidationMeasure {


	@Override
	public Double calculateMeasure(final List<List<double[]>> clusters, final List<double[]> centroids, final boolean erasure) {
		final double stddev = calculateStandardDeviation(clusters, centroids);
		final double interClusterDensity = calculateInterClusterDensity(clusters, centroids, stddev);
		final double intraClusterVariance = calculateIntraClusterVariance(clusters, centroids);
		return interClusterDensity + intraClusterVariance;
	}

	private Double calculateStandardDeviation(final List<List<double[]>> clusters, final List<double[]> centroids) {
		final int dimensions = clusters.get(0).get(0).length;
		final double[] zero = new double[dimensions];
		double sum = 0;
		for (int i = 0; i < clusters.size(); i++) {
			final double[] varianceVector = calculateSigma(clusters.get(i), centroids.get(i));
			sum += singleSquaredEuclideanDistance(zero, varianceVector);
		}
		return Math.sqrt(sum) / clusters.size();
	}

	private double[] calculateSigma(final List<double[]> cluster, final double[] centroid) {
		final int dimensions = centroid.length;

		final double[] variance = new double[dimensions];
		for (int j = 0; j < cluster.size(); j++) {
			for (int k = 0; k < dimensions; k++) {
				variance[k] += Math.pow(centroid[k] - cluster.get(j)[k], 2);
			}

		}
		for (int i = 0; i < dimensions; i++) {
			variance[i] = variance[i] / cluster.size();
		}
		return variance;
	}

	private Double calculateInterClusterDensity(final List<List<double[]>> clusters, final List<double[]> centroids, final double stddev) {
		final double c = clusters.size();
		double dens_bw = 0;
		for (int i = 0; i < c; i++) {
			for (int j = 0; j < i; j++) {
				final double[] vi = centroids.get(i);
				final double[] vj = centroids.get(j);
				final double[] uij = new double[vi.length];
				for (int k = 0; k < vi.length; k++) {
					uij[k] = (vi[k] + vj[k]) / 2;
				}
				final List<double[]> union = new ArrayList<>();
				union.addAll(clusters.get(i));
				union.addAll(clusters.get(j));
				final double densityUnion = calculateDensity(uij, union, stddev);
				final double densityI = calculateDensity(vi, clusters.get(i), stddev);
				final double densityJ = calculateDensity(vj, clusters.get(j), stddev);
				final double max = Math.max(densityI, densityJ);
				if (max != 0) {
					dens_bw += densityUnion / max;
				}
			}
		}
		return dens_bw / (c * (c - 1));
	}

	private double calculateDensity(final double[] point, final List<double[]> neighbourhood, final double stddev) {
		double sum = 0;
		for (int i = 0; i < neighbourhood.size(); i++) {
			if (Math.sqrt(singleSquaredEuclideanDistance(point, neighbourhood.get(i))) < stddev) {
				sum++;
			}
		}
		return sum;
	}

	private Double calculateIntraClusterVariance(final List<List<double[]>> clusters, final List<double[]> centroids) {
		final int dimensions = clusters.get(0).get(0).length;
		final double[] datasetCentroid = new double[dimensions];
		int instances = 0;
		for (int j = 0; j < clusters.size(); j++) {
			for (int i = 0; i < datasetCentroid.length; i++) {
				datasetCentroid[i] += clusters.get(j).size() * centroids.get(j)[i];

			}
			instances += clusters.size();
		}
		for (int i = 0; i < dimensions; i++) {
			datasetCentroid[i] /= instances;
		}

		final List<double[]> sigmas = new ArrayList<>();
		for (int i = 0; i < clusters.size(); i++) {
			sigmas.add(calculateSigma(clusters.get(i), centroids.get(i)));
		}

		final List<double[]> dataset = new ArrayList<>();
		for (int i = 0; i < clusters.size(); i++) {
			dataset.addAll(clusters.get(i));
		}
		final double[] datasetSigma = calculateSigma(dataset, datasetCentroid);

		final double[] zero = new double[dimensions];

		final ArrayList<Double> clusterVariances = new ArrayList<Double>();
		for (int i = 0; i < sigmas.size(); i++) {
			clusterVariances.add(singleSquaredEuclideanDistance(zero, sigmas.get(i)));
		}

		final double datasetVariance = singleSquaredEuclideanDistance(zero, datasetSigma);

		double sum = 0;
		for (int i = 0; i < clusterVariances.size(); i++) {
			sum += clusterVariances.get(i);
		}

		return (sum / clusters.size()) / datasetVariance;
	}

}
