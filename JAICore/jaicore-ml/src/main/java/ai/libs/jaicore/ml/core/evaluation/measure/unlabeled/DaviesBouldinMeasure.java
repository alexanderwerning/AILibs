package ai.libs.jaicore.ml.core.evaluation.measure.unlabeled;


import jaicore.basic.aggregate.IAggregateFunction;
import jaicore.ml.core.dataset.IDataset;
import jaicore.ml.core.dataset.IInstance;
import jaicore.ml.core.evaluation.measure.IMeasure;
import java.util.LinkedList;
import java.util.List;

public class DaviesBouldinMeasure extends AInternalClusteringMeasure {
// In the original paper this is formulated as a loss (minimized for best clustering)
	@Override
	public Double calculateMeasure(List<List<double[]>> clusters) {

		int N = clusters.size();
		List<double[]> centroids = new LinkedList<>();
		for (int i = 0; i < N; i++) {
			centroids.add(calculateCentroid(clusters.get(i)));
		}
		double sum = 0;
		for (int i = 0; i < N; i++) {
			double max = 0;
			for (int j = 0; j < N; j++) {
				if(i!=j){
					double rij = (calculateDispersion(clusters.get(i), centroids.get(i))+calculateDispersion(clusters.get(j), centroids.get(j)))/clusterDistances(clusters.get(i),
						centroids.get(i), clusters.get(j), centroids.get(j));
					if(rij > max){
						max = rij;
					}
				}
			}
			sum += max;
		}
		return sum / N;
	}

	private double clusterDistances(final List<double[]> cluster1, double[] centroid1, final List<double[]> cluster2, double[] centroid2) {
		return distance(centroid1,centroid2);
	}



	private double calculateDispersion(final List<double[]> cluster, double[] centroid) {
		// calculate average distance from centroid
		double sqsum = 0;
		for (int i = 0; i < cluster.size(); i++) {
			double vectorLength = 0;
			for (int j = 0; j < centroid.length; j++) {
				vectorLength += Math.pow(centroid[j]-cluster.get(i)[j],2);
			}
			vectorLength = Math.sqrt(vectorLength);
			sqsum += Math.pow(vectorLength,2);
		}
		sqsum /= cluster.size();
		return Math.sqrt(sqsum);
	}
}
