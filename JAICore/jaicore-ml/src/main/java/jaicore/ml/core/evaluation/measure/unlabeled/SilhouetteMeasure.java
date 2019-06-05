package jaicore.ml.core.evaluation.measure.unlabeled;

import jaicore.basic.aggregate.IAggregateFunction;
import java.util.List;
import weka.core.Instances;

public class SilhouetteMeasure extends AInternalClusteringMeasure {

	@Override
	public Double calculateMeasure(List<List<double[]>> clusters) {
		double sum = 0;
		double size = 0;
		for (int i = 0; i < clusters.size(); i++) {
			List<double[]> cluster = clusters.get(i);
			size += cluster.size();
			for (int j = 0; j < cluster.size(); j++) {
				double[] point = cluster.get(j);
				double distanceToOwnCluster = distancePointToCluster(cluster, point);
				double distanceToOtherCluster = Double.MAX_VALUE;
				for (int k = 0; k < clusters.size(); k++) {
					if(k == i){
						continue;
					}
					double tmp = distancePointToCluster(cluster, point);
					if(tmp > distanceToOtherCluster){
						distanceToOtherCluster = tmp;
					}
				}
				if (distanceToOtherCluster != 0 && distanceToOwnCluster != 0) {
					sum += (distanceToOtherCluster-distanceToOwnCluster)/Math.max(distanceToOtherCluster,distanceToOwnCluster);
				}
			}
		}
		return sum/size;
	}

	private double distancePointToCluster(List<double[]> cluster, double[] point){
		double sum = 0;
		for (int i = 0; i < cluster.size(); i++) {
			sum += distance(cluster.get(i),point);
		}
		return sum/cluster.size();
	}

}
