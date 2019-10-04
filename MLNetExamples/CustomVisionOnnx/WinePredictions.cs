using Microsoft.ML.Data;

namespace CustomVisionOnnx
{
    public class WinePredictions
    {
        [ColumnName("model_outputs0")]
        public float[] PredictedLabels { get; set; }
    }
}
