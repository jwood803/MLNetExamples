using Microsoft.ML.Data;

namespace AzureFunction
{
    public class HousingPrediction
    {
        [ColumnName("Score")]
        public float PredictedPrice { get; set; }
    }
}
