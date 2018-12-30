using Microsoft.ML;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Transforms;

namespace BinaryClassification
{
    class Program
    {
        static void Main(string[] args)
        {
            var context = new MLContext();

            var reader = TextLoader.CreateReader(context, 
                ctx => (
                    PassengerId: ctx.LoadFloat(0),
                    HasSurvived: ctx.LoadBool(1),
                    PClass: ctx.LoadFloat(2),
                    Name: ctx.LoadText(3),
                    Gender: ctx.LoadText(4),
                    Age: ctx.LoadFloat(5),
                    NumOfSiblingsOrSpouses: ctx.LoadFloat(6),
                    NumOfParentOrChildAboard: ctx.LoadFloat(7),
                    Ticket: ctx.LoadText(8),
                    Fare: ctx.LoadFloat(9),
                    Cabin: ctx.LoadText(10),
                    EmbarkedPort: ctx.LoadText(11)
                ),
                hasHeader: true,
                separator: ',');

            var data = reader.Read("titanic.csv");

            var (trainData, testData) = context.BinaryClassification.TrainTestSplit(data, testFraction: 0.2);

            var pipeline = reader.MakeNewEstimator()
                .Append(row => (
                    Target: row.HasSurvived,
                    Features: row.NumOfParentOrChildAboard.ConcatWith(
                        row.NumOfSiblingsOrSpouses, row.PClass, row.Age.ReplaceNaNValues(MissingValueReplacingTransformer.ColumnInfo.ReplacementMode.Mean)
                )));

            pipeline.Append(row => (
                    row.Target,
                    Prediction: context.BinaryClassification.Trainers.LogisticRegression("HasSurvived", "Features")
                ));

            var model = pipeline.Fit(trainData).AsDynamic;

            var predictionFunction = model.MakePredictionFunction<TitanicData, TitanicPrediction>(context);

            var prediction = predictionFunction.Predict(new TitanicData { Gender = "F" });
        }
    }
}
