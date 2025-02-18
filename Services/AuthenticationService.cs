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
                        JsonSerializer.Deserialize<List<double>>(user.FirstH),
                        JsonSerializer.Deserialize<List<double>>(user.SecondH),
                        JsonSerializer.Deserialize<List<double>>(user.ThirdH),
                    };

            var duUserProfile = new List<List<double>>()
                    {
                        JsonSerializer.Deserialize<List<double>>(user.FirstUD),
                        JsonSerializer.Deserialize<List<double>>(user.SecondUD),
                        JsonSerializer.Deserialize<List<double>>(user.ThirdUD),
                    };

            var hUserMedian = Calculations.CalculateMedianValue(hUserProfile);
            var udUserMedian = Calculations.CalculateMedianValue(duUserProfile);

            var normalizedManhattanMethod = AuthenticationMethodFactory.GetAuthenticationMethod(
                Method.NormalizedManhattan,
                hUserMedian,
                udUserMedian,
                hUserProfile,
                duUserProfile);

            return normalizedManhattanMethod.Authenticate(1, loginKeyPressedTimes, loginBetweenKeysTimes);
        }
    }
}
