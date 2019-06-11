package ai.libs.jaicore.ml.dyadranking.optimizing;

import java.util.Map;

import ai.libs.jaicore.math.linearalgebra.Vector;
import ai.libs.jaicore.ml.core.optimizing.IGradientFunction;
import ai.libs.jaicore.ml.dyadranking.Dyad;
import ai.libs.jaicore.ml.dyadranking.dataset.DyadRankingDataset;
import ai.libs.jaicore.ml.dyadranking.dataset.IDyadRankingInstance;

/**
 * Represents a differentiable function in the context of dyad ranking based on
 * feature transformation Placket-Luce models.
 *
 * @author Helena Graf
 *
 */
public interface IDyadRankingFeatureTransformPLGradientFunction extends IGradientFunction {

	/**
	 * Initialize the function with the given data set and feature transformation
	 * method.
	 *
	 * @param dataset
	 *            the dataset to use
	 * @param featureTransforms
	 *            the pre computed feature transformations
	 */
	void initialize(DyadRankingDataset dataset, Map<IDyadRankingInstance, Map<Dyad, Vector>> featureTransforms);
}
