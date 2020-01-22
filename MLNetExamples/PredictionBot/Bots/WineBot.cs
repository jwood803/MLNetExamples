// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Generated with Bot Builder V4 SDK Template for Visual Studio EchoBot v4.6.2

using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Bot.Builder;
using Microsoft.Bot.Builder.Dialogs;
using Microsoft.Bot.Schema;
using Microsoft.Extensions.ML;

namespace PredictionBot.Bots
{
    public class WineBot<T> : ActivityHandler where T : Dialog
    {
        protected readonly Dialog Dialog;
        protected readonly BotState ConversationState;
        protected readonly BotState UserState;
        protected readonly PredictionEnginePool<HousingData, HousingPrediction> PredictionEnginePool;

        public WineBot(ConversationState conversationState, UserState userState, T dialog, 
            PredictionEnginePool<HousingData, HousingPrediction> predictionEnginePool)
        {
            Dialog = dialog;
            ConversationState = conversationState;
            UserState = userState;
            PredictionEnginePool = predictionEnginePool;
        }

        protected override async Task OnMessageActivityAsync(ITurnContext<IMessageActivity> turnContext, CancellationToken cancellationToken)
        {
            await Dialog.RunAsync(turnContext, ConversationState.CreateProperty<DialogState>(nameof(DialogState)), cancellationToken);
        }

        public override async Task OnTurnAsync(ITurnContext turnContext, CancellationToken cancellationToken = default(CancellationToken))
        {
            await base.OnTurnAsync(turnContext, cancellationToken);

            await ConversationState.SaveChangesAsync(turnContext, force: false, cancellationToken);
            await UserState.SaveChangesAsync(turnContext, force: false, cancellationToken);
        }
    }
}
