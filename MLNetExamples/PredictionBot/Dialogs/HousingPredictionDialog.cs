using Microsoft.Bot.Builder;
using Microsoft.Bot.Builder.Dialogs;
using Microsoft.Extensions.ML;
using System.Threading;
using System.Threading.Tasks;

namespace PredictionBot.Dialogs
{
    public class HousingPredictionDialog : ComponentDialog
    {
        private readonly IStatePropertyAccessor<HousingState> _houseStateAccessor;
        private readonly PredictionEnginePool<HousingData, HousingPrediction> _predictionEnginePool;

        public HousingPredictionDialog(UserState userState, PredictionEnginePool<HousingData, HousingPrediction> predictionEnginePool) 
            : base(nameof(HousingPredictionDialog))
        {
            _houseStateAccessor = userState.CreateProperty<HousingState>("HouseState");
            _predictionEnginePool = predictionEnginePool;

            var steps = new WaterfallStep[]
            {
                LatitudeStepAsync,
                FinishDialogAsync
            };

            AddDialog(new WaterfallDialog(nameof(WaterfallDialog), steps));
            AddDialog(new ChoicePrompt(nameof(ChoicePrompt)));
            AddDialog(new TextPrompt(nameof(TextPrompt)));
            AddDialog(new NumberPrompt<int>(nameof(NumberPrompt<int>)));

            InitialDialogId = nameof(WaterfallDialog);
        }

        private static async Task<DialogTurnResult> LatitudeStepAsync(WaterfallStepContext stepContext, CancellationToken cancellationToken)
        {
            return await stepContext.PromptAsync(nameof(NumberPrompt<int>), new PromptOptions
            {
                Prompt = MessageFactory.Text("What is the latitude?")
            });
        }

        private async Task<DialogTurnResult> LongitudeStepAsync(WaterfallStepContext stepContext, CancellationToken cancellationToken)
        {
            stepContext.Values["Latitude"] = stepContext.Result;

            var userProfile = await _houseStateAccessor.GetAsync(stepContext.Context, () => new HousingState(), cancellationToken);

            return await stepContext.PromptAsync(nameof(NumberPrompt<int>), new PromptOptions
            {
                Prompt = MessageFactory.Text("What is the longitude?")
            });
        }

        private static async Task<DialogTurnResult> HousingMedianAgeStepAsync(WaterfallStepContext stepContext, CancellationToken cancellationToken)
        {
            stepContext.Values["Longitude"] = stepContext.Result;

            return await stepContext.PromptAsync(nameof(NumberPrompt<int>), new PromptOptions
            {
                Prompt = MessageFactory.Text("What is the housing median age?")
            });
        }

        private static async Task<DialogTurnResult> TotalRoomsStepAsync(WaterfallStepContext stepContext, CancellationToken cancellationToken)
        {
            stepContext.Values["MedianAge"] = stepContext.Result;

            return await stepContext.PromptAsync(nameof(NumberPrompt<int>), new PromptOptions
            {
                Prompt = MessageFactory.Text("What is the number of total rooms?")
            });
        }

        private static async Task<DialogTurnResult> TotalBedroomsStepAsync(WaterfallStepContext stepContext, CancellationToken cancellationToken)
        {
            stepContext.Values["TotalRooms"] = stepContext.Result;

            return await stepContext.PromptAsync(nameof(NumberPrompt<int>), new PromptOptions
            {
                Prompt = MessageFactory.Text("What is the number of total bedrooms?")
            });
        }

        private static async Task<DialogTurnResult> PopulationStepAsync(WaterfallStepContext stepContext, CancellationToken cancellationToken)
        {
            stepContext.Values["TotalBedrooms"] = stepContext.Result;

            return await stepContext.PromptAsync(nameof(NumberPrompt<int>), new PromptOptions
            {
                Prompt = MessageFactory.Text("What is the population of the area?")
            });
        }

        private static async Task<DialogTurnResult> HouseholdsStepAsync(WaterfallStepContext stepContext, CancellationToken cancellationToken)
        {
            stepContext.Values["Population"] = stepContext.Result;

            return await stepContext.PromptAsync(nameof(NumberPrompt<int>), new PromptOptions
            {
                Prompt = MessageFactory.Text("What is the number of total bedrooms?")
            });
        }

        private async Task<DialogTurnResult> FinishDialogAsync(WaterfallStepContext stepContext, CancellationToken cancellationToken)
        {
            var housingData = new HousingData
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
            };

            var prediction = _predictionEnginePool.Predict(housingData);

            return await stepContext.EndDialogAsync(stepContext.Values, cancellationToken: cancellationToken);
        }
    }
}
