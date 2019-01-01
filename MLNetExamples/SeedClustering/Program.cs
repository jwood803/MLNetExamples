using Microsoft.ML;
using Microsoft.ML.Runtime.Data;
using System;

namespace SeedClustering
{
    class Program
    {
        static void Main(string[] args)
        {
            var dataLocation = "./Seed_Data.csv";

            var context = new MLContext(seed: 42);

            var textLoader = context.Data.TextReader(new TextLoader.Arguments
            {
                Separator = ",",
                HasHeader = true,
                Column = new[]
                {
                    new TextLoader.Column("A", DataKind.R4, 0),
                    new TextLoader.Column("P", DataKind.R4, 1),
                    new TextLoader.Column("C", DataKind.R4, 2),
                    new TextLoader.Column("LK", DataKind.R4, 3),
                    new TextLoader.Column("WK", DataKind.R4, 4),
                    new TextLoader.Column("A_Coef", DataKind.R4, 5),
                    new TextLoader.Column("LKG", DataKind.R4, 6),
                    new TextLoader.Column("Label", DataKind.R4, 7)
                }
            });

            IDataView data = textLoader.Read(dataLocation);

            var (trainData, testData) = context.Clustering.TrainTestSplit(data, testFraction: 0.2);

            var pipeline = context.Transforms.Concatenate("Features", "A", "P", "C", "LK", "WK", "A_Coef", "LKG")
                .Append(context.Clustering.Trainers.KMeans(features: "Features", clustersCount: 3));

            var model = pipeline.Fit(trainData);

            var predictions = model.Transform(testData);

            var metrics = context.Clustering.Evaluate(predictions, score: "Score", features: "Features");

            Console.WriteLine($"Average minimum score: {metrics.AvgMinScore}");

            var predictionFunc = model.MakePredictionFunction<SeedData, SeedPrediction>(context);

            var prediction = predictionFunc.Predict(new SeedData
            {
                A = 13.89F,
                P = 15.33F,
                C = 0.862F,
                LK = 5.42F,
                WK = 3.311F,
                A_Coef = 2.8F,
                LKG = 5
            });

            Console.WriteLine($"Prediction - {prediction.SelectedClusterId}");
        }
    }
}
