using DigitalHandwriting.Factories.AuthenticationMethods;
using DigitalHandwriting.Factories.AuthenticationMethods.Models;
using DigitalHandwriting.Helpers;
using DigitalHandwriting.Models;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

namespace DigitalHandwriting.Services
{
    public class HandwritingAuthenticationResult
    {
        public double KeyPressedDistance { get; set; }

        public double BetweenKeysDistance { get; set; }

        public double BetweenKeysPressDistance { get; set; }

        public double AuthenticationResult { get; set; }
    }

    public static class AuthenticationService
    {
        public static bool PasswordAuthentication(User user, string password)
        {
            var userPasswordHash = user.Password;
            var userPasswordSalt = user.Salt;

            var inputPasswordHash = EncryptionService.GetPasswordHash(password, userPasswordSalt);

            return userPasswordHash.Equals(inputPasswordHash);
        }

        public static AuthenticationResult HandwritingAuthentication(
            User user, 
            List<double> loginKeyPressedTimes, 
            List<double> loginBetweenKeysTimes)
        {
            var hUserProfile = new List<List<double>>()
                    {
                        JsonSerializer.Deserialize<List<double>>(user.KeyPressedTimesFirst),
                        JsonSerializer.Deserialize<List<double>>(user.KeyPressedTimesSecond),
                        JsonSerializer.Deserialize<List<double>>(user.KeyPressedTimesThird),
                    };

            var udUserProfile = new List<List<double>>()
                    {
                        JsonSerializer.Deserialize<List<double>>(user.BetweenKeysPressTimesFirst),
                        JsonSerializer.Deserialize<List<double>>(user.BetweenKeysPressTimesSecond),
                        JsonSerializer.Deserialize<List<double>>(user.BetweenKeysPressTimesThird),
                    };

            var hUserMedian = Calculations.CalculateMedianValue(hUserProfile);
            var udUserMedian = Calculations.CalculateMedianValue(udUserProfile);

            var normalizedManhattanMethod = AuthenticationMethodFactory.GetAuthenticationMethod(
                Method.NormalizedManhattan,
                hUserMedian,
                udUserMedian,
                hUserProfile,
                udUserProfile);

            return normalizedManhattanMethod.Authenticate(1, loginKeyPressedTimes, loginBetweenKeysTimes);
        }
    }
}
