using Microsoft.Extensions.Configuration;
using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Data;
using System.Data.Common;
using System.Data.SqlClient;
using System.IO;
using System.Linq;

namespace DbLoader
{
    class Program
    {
        static void Main(string[] args)
        {
            var builder = new ConfigurationBuilder()
                .SetBasePath(Directory.GetCurrentDirectory())
                .AddJsonFile("config.json");

            var configuration = builder.Build();
            var connectionString = configuration["connectionString"];

            var loaderColumns = new DatabaseLoader.Column[]
            {
                new DatabaseLoader.Column() { Name = "Longitude", Type = DbType.Single },
                new DatabaseLoader.Column() { Name = "Latitude", Type = DbType.Single },
                new DatabaseLoader.Column() { Name = "HousingMedianAge", Type = DbType.Single },
                new DatabaseLoader.Column() { Name = "TotalRooms", Type = DbType.Single },
                new DatabaseLoader.Column() { Name = "TotalBedrooms", Type = DbType.Single },
                new DatabaseLoader.Column() { Name = "Population", Type = DbType.Single },
                new DatabaseLoader.Column() { Name = "Households", Type = DbType.Single },
                new DatabaseLoader.Column() { Name = "MedianIncome", Type = DbType.Single },
                new DatabaseLoader.Column() { Name = "MedianHouseValue", Type = DbType.Single },
                new DatabaseLoader.Column() { Name = "OceanProximity", Type = DbType.String }
            };

            var connection = new SqlConnection(connectionString);
            var factory = DbProviderFactories.GetFactory(connection);

            var context = new MLContext();

            var loader = context.Data.CreateDatabaseLoader(loaderColumns);

            var dbSource = new DatabaseSource(factory, connectionString,
                "SELECT * FROM salarydb.dbo.Housing");

            var data = loader.Load(dbSource);

            var preview = data.Preview();

            var testTrainSplit = context.Data.TrainTestSplit(data, testFraction: 0.2);

            var features = data.Schema
                    .Select(col => col.Name)
                    .Where(colName => colName != "MedianHouseValue" && colName != "OceanProximity")
                    .ToArray();

            var pipeline = context.Transforms.Text.FeaturizeText("Text", "OceanProximity")
                .Append(context.Transforms.Concatenate("Features", features))
                .Append(context.Transforms.Concatenate("Features", "Text"))
                .Append(context.Regression.Trainers.LbfgsPoissonRegression(featureColumnName: "Features", labelColumnName: "MedianHouseValue"));

            var model = pipeline.Fit(testTrainSplit.TestSet);

            var predictionFunc = context.Model.CreatePredictionEngine<HousingData, HousingPrediction>(model);

            var prediction = predictionFunc.Predict(new HousingData
            {
                Longitude = -122.25f,
                Latitude = 37.85f,
                HousingMedianAge = 55.0f,
                TotalRooms = 1627.0f,
                TotalBedrooms = 235.0f,
                Population = 322.0f,
                Households = 120.0f,
                MedianIncome = 8.3014f,
                OceanProximity = "NEAR BAY"
            });

            Console.WriteLine($"Prediction - {prediction.PredictedHouseValue}");

            Console.ReadLine();
        }
    }
}
