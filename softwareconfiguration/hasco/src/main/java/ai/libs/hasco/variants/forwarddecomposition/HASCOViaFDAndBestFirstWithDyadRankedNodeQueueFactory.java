package ai.libs.hasco.variants.forwarddecomposition;

import ai.libs.jaicore.planning.hierarchical.algorithms.forwarddecomposition.graphgenerators.tfd.TFDNode;
import ai.libs.jaicore.search.algorithms.standard.bestfirst.DyadRankedBestFirstFactory;
import ai.libs.jaicore.search.algorithms.standard.bestfirst.IBestFirstQueueConfiguration;
import ai.libs.jaicore.search.algorithms.standard.bestfirst.nodeevaluation.INodeEvaluator;
import ai.libs.jaicore.search.probleminputs.GraphSearchWithSubpathEvaluationsInput;
import ai.libs.jaicore.search.problemtransformers.GraphSearchProblemInputToGraphSearchWithSubpathEvaluationInputTransformer;

/**
 * HASCO variant factory using best first and a dyad-ranked OPEN list.
 * 
 * @author Helena Graf
 *
 */
public class HASCOViaFDAndBestFirstWithDyadRankedNodeQueueFactory extends HASCOViaFDAndBestFirstFactory<Double> {

	/**
	 * Constructs a new HASCO factory with a dyad ranked OPEN list configured with
	 * the given parameters.
	 * 
	 * @param openConfig
	 */
	public HASCOViaFDAndBestFirstWithDyadRankedNodeQueueFactory(
			IBestFirstQueueConfiguration<GraphSearchWithSubpathEvaluationsInput<TFDNode, String, Double>, TFDNode, String, Double> openConfig) {
		super();
		this.setNodeEvaluator(n -> 1.0);
		this.setSearchFactory(new DyadRankedBestFirstFactory<>(openConfig));
	}

	@Override
	public void setNodeEvaluator(INodeEvaluator<TFDNode, Double> nodeEvaluator) {
		setSearchProblemTransformer(
				new GraphSearchProblemInputToGraphSearchWithSubpathEvaluationInputTransformer<>(n -> {
					if (n.isGoal()) {
						return nodeEvaluator.f(n);
					} else {
						return 1.0;
					}
				}));
	}

	@Override
	public HASCOViaFDAndBestFirst<Double> getAlgorithm() {
		HASCOViaFDAndBestFirst<Double> hasco = super.getAlgorithm();
		hasco.setCreateComponentInstancesFromNodesInsteadOfPlans(true);
		return hasco;
	}
}
