using DigitalHandwriting.Helpers;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DigitalHandwriting.Factories.AuthenticationMethods.Models
{
    internal class ITADAuthenticationMethod : AuthenticationMethod
    {
        public ITADAuthenticationMethod(List<double> userKeyPressedTimes, List<double> userBetweenKeysTimes, List<List<double>> userKeyPressedTimesProfile, List<List<double>> userBetweenKeysTimesProfile) : base(userKeyPressedTimes, userBetweenKeysTimes, userKeyPressedTimesProfile, userBetweenKeysTimesProfile)
        {
        }

        public override bool Authenticate(int n, List<double> loginKeyPressedTimes, List<double> loginBetweenKeysTimes)
        {
            if (n > 1)
            {
                return nGraphAuthentication(n, loginKeyPressedTimes, loginBetweenKeysTimes);
            }

            var keyPressedDistance = Calculations.ITAD(loginKeyPressedTimes, UserKeyPressedTimesProfile);
            var betweenKeysDistance = Calculations.ITAD(loginBetweenKeysTimes, UserBetweenKeysTimesProfile);

            // Вычисляем общий результат аутентификации как среднее значение трех расстояний
            var authResult = (keyPressedDistance + betweenKeysDistance) / 2.0;

            Trace.WriteLine($"Key pressed distance: {keyPressedDistance}, " +
                $"Between keys distance: {betweenKeysDistance}, " +
                $"auth result : {authResult}");

            // Возвращаем true, если результат аутентификации меньше порогового значения
            return authResult > 0.45;
        }

        private bool nGraphAuthentication(int n, List<double> loginKeyPressedTimes, List<double> loginBetweenKeysTimes)
        {
            var userNGraphs = Calculations.CalculateNGraph(n, UserKeyPressedTimesProfile, UserBetweenKeysTimesProfile);
            var loginNGraph = Calculations.CalculateNGraph(n, loginKeyPressedTimes, loginBetweenKeysTimes);

            var sumDistance = 0.0;
            foreach (var key in loginNGraph.Keys)
            {
                var values1 = userNGraphs[key];
                var values2 = loginNGraph[key];

                if (values1[0].Count != values2.Count)
                {
                    throw new ArgumentException("Количество значений для каждой метрики должно быть одинаковым.");
                }

                sumDistance += Calculations.ITAD(values2, values1);
            }

            // Вычисляем общий результат аутентификации как среднее значение трех расстояний
            var authResult = sumDistance / loginNGraph.Keys.Count;

            Trace.WriteLine($"auth result : {authResult}");

            // Возвращаем true, если результат аутентификации меньше порогового значения
            return authResult > 0.45;
        }
    }
}
