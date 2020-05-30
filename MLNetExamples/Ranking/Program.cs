using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;

namespace Ranking
{
    class Program
    {
        static void Main(string[] args)
        {
            var context = new MLContext();

            var data = context.Data.LoadFromTextFile<RankingData>("./ranking.tsv", separatorChar: '\t');

            var split = context.Data.TrainTestSplit(data, testFraction: 0.2);

            var secondSplit = context.Data.TrainTestSplit(split.TestSet);

            //var validation = secondSplit.TrainSet;

            var sampleInput = secondSplit.TestSet;

            var rankingPipeline = context.Transforms.Conversion.MapValueToKey("Label")
                .Append(context.Transforms.Conversion.Hash("GroupId", "GroupId"))
                .Append(context.Ranking.Trainers.LightGbm());

            var model = rankingPipeline.Fit(split.TrainSet);

            var predictions = model.Transform(split.TestSet);

            var options = new RankingEvaluatorOptions
            {
                DcgTruncationLevel = 5
            };

            var metrics = context.Ranking.Evaluate(predictions, options);

            var ndcg = metrics.NormalizedDiscountedCumulativeGains.Average();

            Console.WriteLine($"nDGC - {ndcg}");
            Console.Write(Environment.NewLine);
            
            var batchPredictions = model.Transform(sampleInput);

            var newPredictions = context.Data.CreateEnumerable<RankingPrediction>(batchPredictions, reuseRowObject: false);

            Console.WriteLine("Scores:");
            foreach (var prediction in newPredictions)
            {
                Console.WriteLine($"{prediction.RelevanceScore}");
            }

            //Console.WriteLine($"Relevance - {prediction.Score}");
        }
    }
}
