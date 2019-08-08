using Microsoft.Extensions.Configuration;
using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Data;
using System.Data.Common;
using System.Data.SqlClient;
using System.IO;

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
                new DatabaseLoader.Column() { Name = "YearsOfExperience", Type = DbType.Single },
                new DatabaseLoader.Column() { Name = "Salary", Type = DbType.Single }
            };

            var connection = new SqlConnection(connectionString);
            var factory = DbProviderFactories.GetFactory(connection);

            var context = new MLContext();

            var loader = context.Data.CreateDatabaseLoader(loaderColumns);

            var dbSource = new DatabaseSource(factory, connectionString,
                "SELECT YearsOfExperience, Salary FROM salarydb.dbo.SalaryData");

            var data = loader.Load(dbSource);

            var preview = data.Preview();

            var testTrainData = context.Data.TrainTestSplit(data, testFraction: 0.2);

            var pipeline = context.Transforms.Concatenate("Features", "YearsOfExperience")
                .Append(context.Transforms.CopyColumns("Label", "Salary"))
                .Append(context.Regression.Trainers.Sdca());

            var model = pipeline.Fit(testTrainData.TrainSet);

            var prediction = model.Transform(testTrainData.TestSet);

            var metrics = context.Regression.Evaluate(prediction);
        }
    }
}
