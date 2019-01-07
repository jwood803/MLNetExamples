using Microsoft.ML;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Transforms;
using System;
using System.Linq;

namespace BinaryClassification
{
    class Program
    {
        static void Main(string[] args)
        {
            var context = new MLContext();

            var textLoader = context.Data.TextReader(new TextLoader.Arguments()
            {
                Separator = ",",
                HasHeader = true,
                Column = new[]
                {
                    new TextLoader.Column("PassengerId", DataKind.R4, 0),
                    new TextLoader.Column("Label", DataKind.Bool, 1),
                    new TextLoader.Column("Pclass", DataKind.R4, 2),
                    new TextLoader.Column("Name", DataKind.Text, 3),
                    new TextLoader.Column("Sex", DataKind.Text, 4),
                    new TextLoader.Column("Age", DataKind.R4, 5),
                    new TextLoader.Column("SibSp", DataKind.R4, 6),
                    new TextLoader.Column("Parch", DataKind.R4, 7),
                    new TextLoader.Column("Ticket", DataKind.Text, 8),
                    new TextLoader.Column("Fare", DataKind.R4, 9),
                    new TextLoader.Column("Cabin", DataKind.Text, 10),
                    new TextLoader.Column("Embarked", DataKind.Text, 11)
                }
            });

            IDataView data = textLoader.Read("titanic.csv");

            var (trainData, testData) = context.BinaryClassification.TrainTestSplit(data, testFraction: 0.2);

            var pipeline = context.Transforms.Concatenate("Text", "Name", "Sex", "Embarked")
                .Append(context.Transforms.Text.FeaturizeText("Text", "TextFeatures"))
                .Append(context.Transforms.Concatenate("Features", "TextFeatures", "Pclass", "Age", "Fare", "SibSp", "Parch"))
                .Append(context.BinaryClassification.Trainers.LogisticRegression("Label", "Features"));

            Console.WriteLine("Cross validating...");

            var crossValidateResults = context.BinaryClassification.CrossValidate(testData, pipeline);

            var averageAuc = crossValidateResults.Average(i => i.metrics.Auc);

            Console.WriteLine($"Average AUC - {averageAuc}");

            var model = pipeline.Fit(trainData);

            var predictionFunction = model.MakePredictionFunction<TitanicData, TitanicPrediction>(context);

            var prediction = predictionFunction.Predict(new TitanicData { Sex = "F" });

            Console.WriteLine($"Prediction - {prediction.Prediction}");

            Console.ReadLine();
        }
    }
}
