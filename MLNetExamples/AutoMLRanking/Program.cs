using System;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;

namespace AutoMLRanking
{
    class Program
    {
        static void Main(string[] args)
        {
            var context = new MLContext();

            var data = context.Data.LoadFromTextFile<RankingData>("./ranking.tsv", separatorChar: '\t');

            var trainTestSplit = context.Data.TrainTestSplit(data, testFraction: 0.2);

            var settings = new RankingExperimentSettings
            {
                MaxExperimentTimeInSeconds = 300,
                OptimizingMetric = RankingMetric.Ndcg,
            };

            var experiment = context.Auto().CreateRankingExperiment(settings);

            var progressHandler = new Progress<RunDetail<RankingMetrics>>(ph =>
            {
                if (ph.ValidationMetrics != null)
                {
                    Console.WriteLine($"Current trainer - {ph.TrainerName} with nDCG {ph.ValidationMetrics.NormalizedDiscountedCumulativeGains.Average()}");
                }
            });

            var results = experiment.Execute(trainTestSplit.TrainSet, validationData: trainTestSplit.TestSet, 
                progressHandler: progressHandler);

            var bestRun = results.BestRun;

            var metrics = bestRun.ValidationMetrics.NormalizedDiscountedCumulativeGains;

            Console.WriteLine(Environment.NewLine);
            Console.WriteLine($"Best model {bestRun.TrainerName} - with nDCG {metrics.Average()}");
        }
    }
}
