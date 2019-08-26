using Microsoft.ML.Data;

namespace DataPrepRowsColumns
{
    public class HousingPrediction
    {
        [ColumnName("Score")]
        public float PredictedPrice { get; set; }
    }
}
