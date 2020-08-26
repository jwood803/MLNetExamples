using System;
using Microsoft.ML;
using Microsoft.ML.AutoML;

namespace AutoMLRanking
{
    class Program
    {
        static void Main(string[] args)
        {
            var context = new MLContext();

            var data = context.Data.LoadFromTextFile<RankingData>("./ranking.tsv", separatorChar: '\t');

            var settings = new RankingExperimentSettings
            {
                MaxExperimentTimeInSeconds = 300,
                OptimizingMetric = RankingMetric.Ndcg
            };

            var experiment = context.Auto().CreateRankingExperiment(settings);

            var results = experiment.Execute(data);

            var bestModel = results.BestRun.Model;
        }
    }
}
