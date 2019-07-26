package ai.libs.jaicore.ml.core.evaluation.measure.unlabeled;

import java.util.List;

public class VanDongenMeasure extends AExternalClusteringValidationMeasure {

	@Override
	public Double calculateExternalMeasure(final List<Double> actual, final List<Double> expected, final int[][] contingencyMatrix) {
		final double n = 2 * actual.size();
		double result = n;
		for (int i = 0; i < contingencyMatrix.length; i++) {
			double maxi = 0;
			for (int j = 0; j < contingencyMatrix[0].length; j++) {
				if (contingencyMatrix[i][j] > maxi) {
					maxi = contingencyMatrix[i][j];
				}
			}
			result -= maxi;
		}
		for (int j = 0; j < contingencyMatrix[0].length; j++) {
			double maxj = 0;
			for (int i = 0; i < contingencyMatrix.length; i++) {
				if (contingencyMatrix[i][j] > maxj) {
					maxj = contingencyMatrix[i][j];
				}
			}
			result -= maxj;
		}
		return result / (n);
	}
}
