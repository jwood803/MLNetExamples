using Microsoft.Extensions.Configuration;
using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Data.SqlClient;
using System.IO;
using System.Linq;

namespace DatabaseData
{
    class Program
    {
        private static string _connectionString;

        static void Main(string[] args)
        {
            var builder = new ConfigurationBuilder()
                .SetBasePath(Directory.GetCurrentDirectory())
                .AddJsonFile("config.json");

            var configuration = builder.Build();
            _connectionString = configuration["connectionString"];

            var fileData = ReadFromFile("./SalaryData.csv");

            AddDataToDatabase(fileData);

            var dbData = ReadFromDatabase();

            var context = new MLContext();

            var mlData = context.Data.ReadFromEnumerable(dbData);

            var (trainData, testData) = context.Regression.TrainTestSplit(mlData, testFraction: 0.2);

            var preview = trainData.Preview(maxRows: 5);

            var pipeline = context.Transforms.Concatenate("Features", "YearsExperience")
                .Append(context.Transforms.CopyColumns(("Label", "Salary")))
                .Append(context.Regression.Trainers.FastTree());

            var model = pipeline.Fit(trainData);

            var prediction = model.Transform(testData);

            var metrics = context.Regression.Evaluate(prediction);

            var predictionFunc = model.CreatePredictionEngine<SalaryData, SalaryPrediction>(context);

            var salaryPrediction = predictionFunc.Predict(new SalaryData { YearsExperience = 11 });

            Console.WriteLine($"Prediction - {salaryPrediction.PredictedSalary}");
            Console.ReadLine();
        }

        private static IEnumerable<SalaryData> ReadFromDatabase()
        {
            var data = new List<SalaryData>();

            using (var conn = new SqlConnection(_connectionString))
            {
                conn.Open();

                var selectCommand = "SELECT YearsOfExperience, Salary FROM mlnetExample.dbo.SalaryData";

                var sqlCommand = new SqlCommand(selectCommand, conn);

                var reader = sqlCommand.ExecuteReader();

                while(reader.Read())
                {
                    data.Add(new SalaryData
                    {
                        YearsExperience = float.Parse(reader.GetValue(0).ToString()),
                        Salary = float.Parse(reader.GetValue(1).ToString())
                    });
                }
            }

            return data;
        }

        private static void AddDataToDatabase(IEnumerable<SalaryData> data)
        {
            using (var conn = new SqlConnection(_connectionString))
            {
                conn.Open();

                var insertCommand = "INSERT INTO mlnetExample.dbo.SalaryData VALUES (@years, @salary);";
                var selectCommand = "SELECT COUNT(*) From mlnetExample.dbo.SalaryData";

                var selectSqlCommand = new SqlCommand(selectCommand, conn);

                var results = (int)selectSqlCommand.ExecuteScalar();

                if (results > 0)
                {
                    var deleteCommand = "DELETE FROM mlnetExample.dbo.SalaryData";

                    var deleteSqlCommand = new SqlCommand(deleteCommand, conn);

                    deleteSqlCommand.ExecuteNonQuery();
                }

                foreach (var item in data)
                {
                    var command = new SqlCommand(insertCommand, conn);

                    command.Parameters.AddWithValue("@years", item.YearsExperience);
                    command.Parameters.AddWithValue("@salary", item.Salary);

                    command.ExecuteNonQuery();
                }
            }
        }

        private static IEnumerable<SalaryData> ReadFromFile(string filePath)
        {
            var data = File.ReadAllLines(filePath)
                .Skip(1)
                .Select(l => l.Split(','))
                .Select(i => new SalaryData
                {
                    YearsExperience = float.Parse(i[0]),
                    Salary = float.Parse(i[1])
                });

            return data;
        }
    }
}
