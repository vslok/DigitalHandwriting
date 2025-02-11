using DigitalHandwriting.Helpers;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DigitalHandwriting.Factories.AuthenticationMethods.Models
{
    class EuclidianAuthenticationMethod : AuthenticationMethod
    {
        public EuclidianAuthenticationMethod(List<double> userKeyPressedTimes, List<double> userBetweenKeysTimes, List<List<double>> userKeyPressedTimesProfile, List<List<double>> userBetweenKeysTimesProfile) : base(userKeyPressedTimes, userBetweenKeysTimes, userKeyPressedTimesProfile, userBetweenKeysTimesProfile)
        {
        }

        public override bool Authenticate(int n, List<double> loginKeyPressedTimes, List<double> loginBetweenKeysTimes)
        {
            if (n > 1)
            {
                return nGraphAuthentication(n, loginKeyPressedTimes, loginBetweenKeysTimes);
            }

            var keyPressedDistance = Calculations.EuclideanDistance(UserKeyPressedTimes, loginKeyPressedTimes);
            var betweenKeysDistance = Calculations.EuclideanDistance(UserBetweenKeysTimes, loginBetweenKeysTimes);

            // Вычисляем общий результат аутентификации как среднее значение трех расстояний
            var authResult = (keyPressedDistance + betweenKeysDistance) / 2.0;

            Trace.WriteLine($"Key pressed distance: {keyPressedDistance}, " +
                $"Between keys distance: {betweenKeysDistance}, " +
                $"auth result : {authResult}");

            // Возвращаем true, если результат аутентификации меньше порогового значения
            return authResult < 0.15;
        }

        private bool nGraphAuthentication(int n, List<double> loginKeyPressedTimes, List<double> loginBetweenKeysTimes)
        {
            var userNGraph = Calculations.CalculateNGraph(n, UserKeyPressedTimes, UserBetweenKeysTimes);
            var loginNGraph = Calculations.CalculateNGraph(n, loginKeyPressedTimes, loginBetweenKeysTimes);

            var sumDistance = 0.0;
            foreach (var key in userNGraph.Keys)
            {
                var values1 = userNGraph[key];
                var values2 = loginNGraph[key];

                if (values1.Count != values2.Count)
                {
                    throw new ArgumentException("Количество значений для каждой метрики должно быть одинаковым.");
                }

                sumDistance += Calculations.EuclideanDistance(values1, values2);
            }

            // Вычисляем общий результат аутентификации как среднее значение трех расстояний
            var authResult = sumDistance / userNGraph.Keys.Count;

            Trace.WriteLine($"auth result : {authResult}");

            // Возвращаем true, если результат аутентификации меньше порогового значения
            return authResult < 0.15;
        }
    }
}
