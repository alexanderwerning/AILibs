package ai.libs.mlplan;

import ai.libs.hasco.model.Component;
import ai.libs.hasco.model.ComponentInstance;
import ai.libs.hasco.model.ComponentUtil;
import ai.libs.hasco.model.NumericParameterDomain;
import ai.libs.hasco.model.Parameter;
import ai.libs.hasco.serialization.ComponentLoader;
import ai.libs.jaicore.basic.SQLAdapter;
import ai.libs.jaicore.basic.TimeOut;
import ai.libs.jaicore.basic.sets.PartialOrderedSet;
import ai.libs.jaicore.experiments.Experiment;
import ai.libs.jaicore.experiments.ExperimentDBEntry;
import ai.libs.jaicore.experiments.ExperimentRunner;
import ai.libs.jaicore.experiments.IExperimentIntermediateResultProcessor;
import ai.libs.jaicore.experiments.IExperimentSetEvaluator;
import ai.libs.jaicore.experiments.databasehandle.ExperimenterSQLHandle;
import ai.libs.jaicore.experiments.exceptions.ExperimentDBInteractionFailedException;
import ai.libs.jaicore.experiments.exceptions.ExperimentEvaluationFailedException;
import ai.libs.jaicore.experiments.exceptions.IllegalExperimentSetupException;
import ai.libs.jaicore.ml.core.evaluation.measure.unlabeled.DaviesBouldinMeasure;
import ai.libs.jaicore.ml.core.evaluation.measure.unlabeled.DensityBasedMeasure;
import ai.libs.jaicore.ml.core.evaluation.measure.unlabeled.DunnMeasure;
import ai.libs.jaicore.ml.core.evaluation.measure.unlabeled.IInternalClusteringValidationMeasure;
import ai.libs.jaicore.ml.core.evaluation.measure.unlabeled.SilhouetteMeasure;
import ai.libs.jaicore.ml.core.evaluation.measure.unlabeled.SquaredErrorMeasure;
import ai.libs.mlplan.ClusteringEvaluation.ClusteringResult;
import ai.libs.mlplan.multiclass.wekamlplan.sklearn.SKLearnClassifierFactory;
import ai.libs.mlplan.multiclass.wekamlplan.sklearn.SKLearnClusterClassifierFactory;
import java.io.File;
import java.io.FileReader;
import java.sql.SQLException;
import java.util.Collection;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Timer;
import java.util.TimerTask;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;
import org.aeonbits.owner.ConfigFactory;
import org.apache.commons.lang3.exception.ExceptionUtils;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NominalToBinary;

public class RandomClusterSearch implements IExperimentSetEvaluator {

	private static final IAutoMLForClusteringExperimentConfig CONFIG = ConfigFactory.create(IAutoMLForClusteringExperimentConfig.class);
	private static final AtomicInteger status = new AtomicInteger(1);
	private static final String FS_SEARCH_SPACE_CONFIG = "softwareconfiguration/mlplan/resources/automl/searchmodels/sklearn/sklearn-cluster-mlplan.json";
	private static final String DEF_REQUESTED_HASCO_INTERFACE = "ClusteringAlgorithm";

	@Override
	public void evaluate(final ExperimentDBEntry experimentEntry, final IExperimentIntermediateResultProcessor processor) throws ExperimentEvaluationFailedException {
		try {
			System.out.println("ExperimentEntry: " + experimentEntry);
			final Experiment experiment = experimentEntry.getExperiment();
			System.out.println("Conduct experiment " + experiment);
			final Map<String, String> experimentDescription = experiment.getValuesOfKeyFields();
			System.out.println("Experiment Description as follows: " + experimentDescription);
			final int numCPUs = experiment.getNumCPUs();
			final ExecutorService executorService = Executors.newFixedThreadPool(numCPUs);

			try (final SQLAdapter adapter = new SQLAdapter(CONFIG.getDBHost(), CONFIG.getDBUsername(), CONFIG.getDBPassword(), CONFIG.getDBDatabaseName())) {
				final int seed = Integer.parseInt(experimentDescription.get("seed"));
				final Random rand = new Random(seed);
				final FileReader fileReader = new FileReader(new File(CONFIG.datasetDirectory(), experimentDescription.get("dataset") + ".arff"));
				final Instances unprocessedData = new Instances(fileReader);
				// ScikitLearnWrapper does not support nominal attributes, the NominalToNumeric Filter is used to convert the dataset
				//TODO MISSING DATA? DELETE ENTIRE ROW?
				unprocessedData.removeIf(Instance::hasMissingValue);
				if (unprocessedData.size() == 0) {
					System.out.println("size is Zero");
					return;
				}
				final Filter filter = new NominalToBinary();
				//filter.setOptions(new String[]{"-N"});
				filter.setInputFormat(unprocessedData);
				filter.batchFinished();
				final Instances data = Filter.useFilter(unprocessedData, filter);

				/////////
				final TimeOut candidateTimeout = new TimeOut(Integer.parseInt(experimentDescription.get("candTimeout")), TimeUnit.SECONDS);
				System.out.println("Get list of all possible algorithm selections.");
				final Collection<Component> allComponents = new ComponentLoader(new File(FS_SEARCH_SPACE_CONFIG)).getComponents();
				final Collection<Component> componentsProvidingInterface = ComponentUtil.getComponentsProvidingInterface(allComponents, DEF_REQUESTED_HASCO_INTERFACE);
				//ADAPT HYPERPARAMETERS TO DATASET PROPERTIES
				for (final Component component : componentsProvidingInterface
				) {
					String paramName = "n_clusters";
					switch (component.getName()) {
						case "sklearn.cluster.DBSCAN":
							paramName = "min_samples";
						case "sklearn.cluster.AgglomerativeClustering":
						case "sklearn.cluster.KMeans":
							final PartialOrderedSet<Parameter> parameters = component.getParameters();
							final Parameter nClusters = component.getParameterWithName(paramName);
							parameters.remove(nClusters);
							final NumericParameterDomain domain = (NumericParameterDomain) nClusters.getDefaultDomain();
							final NumericParameterDomain newDomain = new NumericParameterDomain(domain.isInteger(), domain.getMin(), Math.min(domain.getMax(), data.size()));
							final Parameter newNCluster = new Parameter(nClusters.getName(), newDomain, nClusters.getDefaultValue());
							parameters.add(newNCluster);
					}
				}
				final List<ComponentInstance> possibleAlgorithmSelections =
					componentsProvidingInterface.stream().map(x -> new ComponentInstance(x, new HashMap<>(), new HashMap<>())).collect(Collectors.toList());

				System.out.println("Number of possible algorithm selections: " + possibleAlgorithmSelections.size());
				final SKLearnClassifierFactory factory = new SKLearnClusterClassifierFactory();
				final Timer timer = new Timer();
				timer.schedule(new TimerTask() {
					@Override
					public void run() {
						System.out.println("Cancel random search");
						status.decrementAndGet();
						executorService.shutdownNow();
						final Map<String, Object> expResult = new HashMap<>();
						expResult.put("done", "true");
						processor.processResults(expResult);
					}
				}, new TimeOut(Integer.parseInt(experimentDescription.get("timeout")), TimeUnit.SECONDS).milliseconds());
				System.out.println("Schedule jobs for evaluations.");
				final long startTimestamp = System.currentTimeMillis();
				for (int i = 0; i < experimentEntry.getExperiment().getNumCPUs(); i++) {
					executorService.submit(new Runnable() {
						@Override
						public void run() {
							try {
								while (status.get() == 1) {
									if (Thread.interrupted()) {
										return;
									}
									final ComponentInstance ci = possibleAlgorithmSelections.get(rand.nextInt(possibleAlgorithmSelections.size()));
									final LinkedList<ComponentInstance> cisForRandomParams = new LinkedList<>();
									cisForRandomParams.add(ci);
									while (!cisForRandomParams.isEmpty()) {
										final ComponentInstance ciHyp = cisForRandomParams.poll();
										cisForRandomParams.addAll(ciHyp.getSatisfactionOfRequiredInterfaces().values());
										ciHyp.getParameterValues().clear();
										ciHyp.getParameterValues().putAll(ComponentUtil.randomParameterizationOfComponent(ciHyp.getComponent(), rand).getParameterValues());
									}
									final Map<String, Object> results = new HashMap<>();
									results.put("experiment_id", experimentEntry.getId());
									results.put("mainClassifier", ci.getNestedComponentDescription());
									results.put("componentInstance", ci.toString());
									results.put("internalMeasure", experimentDescription.get("internalClusterValidationMeasure"));
									final IInternalClusteringValidationMeasure internalMeasure;
									switch (experimentDescription.get("internalClusterValidationMeasure")) {
										case "DaviesBouldin":
											internalMeasure = new DaviesBouldinMeasure();
											break;
										case "DensityBased":
											internalMeasure = new DensityBasedMeasure();
											break;
										case "Dunn":
											internalMeasure = new DunnMeasure();
											break;
										case "Silhouette":
											internalMeasure = new SilhouetteMeasure();
											break;
										case "SquaredError":
											internalMeasure = new SquaredErrorMeasure();
											break;
										default:
											internalMeasure = new SquaredErrorMeasure();
									}
									final TimerTask candidateTimeoutTask = new TimerTask() {
										private final Thread currentThread = Thread.currentThread();


										@Override
										public void run() {
											System.out.println("Interrupt current thread evaluating " + ci.getNestedComponentDescription());
											this.currentThread.interrupt();
										}
									};
									timer.schedule(candidateTimeoutTask, candidateTimeout.milliseconds());
									System.out.println("Evaluate " + ci.getNestedComponentDescription() + " " + experimentDescription.get("dataset"));
									try {
										final long evalStart;
										{
											final Classifier c = factory.getComponentInstantiation(ci);

											/* Validate the classifier */
											evalStart = System.currentTimeMillis();
											final ClusteringResult valRes = ClusteringEvaluation.evaluateModel(c, data, internalMeasure);
											results.put("valTrainTime", (double) (System.currentTimeMillis() - evalStart));

											results.put("internalMeasureResult", valRes.getInternalEvaluationResult());
										}

										results.put("secondsUntilFound", (int) ((double) (System.currentTimeMillis() - startTimestamp) / 1000));
										results.put("exception", "");
									} catch (final Exception e) {
										e.printStackTrace();
										final StringBuilder stackTraceBuilder = new StringBuilder();
										for (final Throwable ex : ExceptionUtils.getThrowables(e)) {
											stackTraceBuilder.append(ExceptionUtils.getStackTrace(ex) + "\n");
										}
										results.put("exception", stackTraceBuilder.toString());
									} finally {
										candidateTimeoutTask.cancel();
									}
									try {
										adapter.insert("randomsearch_eval", results);
										results.remove("componentInstance");
										System.out.println("Results collected " + results);
									} catch (final SQLException e) {
										e.printStackTrace();
									}

								}
							} catch (final Throwable e) {
								e.printStackTrace();
							}
						}
					});
				}
			}
		} catch (final Exception e) {
			e.printStackTrace();
			throw new ExperimentEvaluationFailedException(e);
		}

	}

	public static void main(final String[] args) throws ExperimentDBInteractionFailedException, IllegalExperimentSetupException {
		System.out.println(CONFIG);
		final ExperimentRunner runner = new ExperimentRunner(CONFIG, new RandomClusterSearch(), new ExperimenterSQLHandle(CONFIG));//, 1);
		runner.randomlyConductExperiments(1, false);
	}

}