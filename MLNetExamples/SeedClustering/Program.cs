using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Data;
using System;

namespace SeedClustering
{
    class Program
    {
        static void Main(string[] args)
        {
            var dataLocation = "./Seed_Data.csv";

            var context = new MLContext();

            var textLoader = context.Data.CreateTextLoader(new[]
            {
                new TextLoader.Column("A", DataKind.Single, 0),
                new TextLoader.Column("P", DataKind.Single, 1),
                new TextLoader.Column("C", DataKind.Single, 2),
                new TextLoader.Column("LK", DataKind.Single, 3),
                new TextLoader.Column("WK", DataKind.Single, 4),
                new TextLoader.Column("A_Coef", DataKind.Single, 5),
                new TextLoader.Column("LKG", DataKind.Single, 6),
                new TextLoader.Column("Label", DataKind.Single, 7)
            },
            hasHeader: true,
            separatorChar: ',');

            IDataView data = textLoader.Load(dataLocation);

            var trainTestData = context.Clustering.TrainTestSplit(data, testFraction: 0.2);

            var pipeline = context.Transforms.Concatenate("Features", "A", "P", "C", "LK", "WK", "A_Coef", "LKG")
                .Append(context.Clustering.Trainers.KMeans(featureColumnName: "Features", clustersCount: 3));

            var preview = trainTestData.TrainSet.Preview();

            var model = pipeline.Fit(trainTestData.TrainSet);

            var predictions = model.Transform(trainTestData.TestSet);

            var metrics = context.Clustering.Evaluate(predictions, score: "Score", features: "Features");

            Console.WriteLine($"Average minimum score: {metrics.AvgMinScore}");

            var predictionFunc = model.CreatePredictionEngine<SeedData, SeedPrediction>(context);

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
            Console.ReadLine();
        }
    }
}
