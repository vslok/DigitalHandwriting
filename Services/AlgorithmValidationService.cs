using CsvHelper; // Keep if WriteResultsToCsv uses it
using DigitalHandwriting.Factories.AuthenticationMethods;
using DigitalHandwriting.Factories.AuthenticationMethods.Models;
using DigitalHandwriting.Helpers;
using DigitalHandwriting.Models;
using DigitalHandwriting.Repositories;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Globalization; // Keep if WriteResultsToCsv uses it
using System.IO; // Keep for Path, Directory
using System.Linq; // Required for .Any()
// System.Text.Json might no longer be needed here if all direct uses are removed
using System.Threading.Tasks; // For Parallel.ForEach

namespace DigitalHandwriting.Services
{
    public class AuthenticationValidationResult : AuthenticationResult
    {
        public AuthenticationValidationResult(AuthenticationResult authenticationResult, string userLogin, bool isLegalUser, Method authenticationMethod) : base(authenticationResult)
        {
            Login = userLogin;
            IsLegalUser = isLegalUser;
            AuthenticationMethod = authenticationMethod;
        }

        public string Login { get; set; }
        public bool IsLegalUser { get; set; }
        public Method AuthenticationMethod { get; set; }
    }

    public class BiometricMetrics // Unchanged
    {
        public double FAR { get; set; } // False Acceptance Rate
        public double FRR { get; set; } // False Rejection Rate
    }

    public class AlgorithmValidationService
    {
        private readonly DataMigrationService _dataMigrationService;
        private readonly UserRepository _userRepository;

        public AlgorithmValidationService()
        {
            _dataMigrationService = new DataMigrationService();
            _userRepository = new UserRepository();
        }

        public async Task ValidateAuthentication(string testFilePath, int n, string saveDirectory) // Made async Task
        {
            var systemUsers = _userRepository.getAllUsers();
            var testAuthentications = _dataMigrationService.GetAuthenticationDataFromCsv(testFilePath);
            var resultsByMethod = new ConcurrentDictionary<Method, ConcurrentBag<AuthenticationValidationResult>>();
            var thresholds = new List<double>() { 0.20 };
            /*for (double t = 0.0; t <= 1.0; t += 0.01)
            {
                thresholds.Add(Math.Round(t, 2));
            }*/

            var processingTasks = new List<Task>();

            foreach (var testAuthRecord in testAuthentications)
            {
                processingTasks.Add(ProcessSingleTestRecordAsync(testAuthRecord, n, systemUsers, resultsByMethod, thresholds));
            }

            await Task.WhenAll(processingTasks);

            foreach (var methodEntry in resultsByMethod)
            {
                WriteResultsToCsv(methodEntry.Value.ToList(), saveDirectory);
            }
        }

        private async Task ProcessSingleTestRecordAsync(
            CsvImportAuthentication testAuthenticationsRecord, // Changed type to CsvImportAuthentication
            int n,
            List<User> systemUsers,
            ConcurrentDictionary<Method, ConcurrentBag<AuthenticationValidationResult>> resultsByMethod,
            List<double> thresholds)
        {
            void ProcessMethod(Method method, List<AuthenticationResult> currentResults, string userLogin, bool isLegalUser)
            {
                foreach (var result in currentResults)
                {
                    var validationResult = new AuthenticationValidationResult(
                        result,
                        userLogin,
                        isLegalUser,
                        method
                    );
                    resultsByMethod
                        .GetOrAdd(method, _ => new ConcurrentBag<AuthenticationValidationResult>())
                        .Add(validationResult);
                }
            }

            var user = systemUsers.Find((u) => u.Login == testAuthenticationsRecord.Login);

            if (user == null)
            {
                Console.WriteLine($"Error: User {testAuthenticationsRecord.Login} not found in system users during validation. Skipping record.");
                return;
            }

            var hUserProfile = user.HSampleValues;
            var udUserProfile = user.UDSampleValues;

            if (hUserProfile == null || udUserProfile == null)
            {
                Console.WriteLine($"Error: User {user.Login} has null HSampleValues or UDSampleValues. Skipping record.");
                return;
            }

            if (!hUserProfile.Any(list => list.Any()) || !udUserProfile.Any(list => list.Any()))
            {
                Console.WriteLine($"Warning: User {user.Login} has HSampleValues or UDSampleValues that are empty or contain only empty sample lists. Median calculation might be affected or fail. Skipping record.");
                return;
            }

            var hUserMedian = Calculations.CalculateMeanValue(hUserProfile);
            var udUserMedian = Calculations.CalculateMeanValue(udUserProfile);

            var authenticationH = new List<double>(testAuthenticationsRecord.H);
            var authenticationDU = new List<double>(testAuthenticationsRecord.UD);

            // --- Existing Methods ---
            var euclidianMethod = AuthenticationMethodFactory.GetAuthenticationMethod(
                Method.Euclidian, hUserMedian, udUserMedian, hUserProfile, udUserProfile);
            var euclidianMethodResult = await euclidianMethod.Authenticate(user.Login, n, authenticationH, authenticationDU, thresholds);
            ProcessMethod(Method.Euclidian, euclidianMethodResult, user.Login, testAuthenticationsRecord.IsLegalUser);

            var euclidianNormalizedMethod = AuthenticationMethodFactory.GetAuthenticationMethod(
                Method.NormalizedEuclidian, hUserMedian, udUserMedian, hUserProfile, udUserProfile);
            var euclidianNormalizedMethodResult = await euclidianNormalizedMethod.Authenticate(user.Login, n, authenticationH, authenticationDU, thresholds);
            ProcessMethod(Method.NormalizedEuclidian, euclidianNormalizedMethodResult, user.Login, testAuthenticationsRecord.IsLegalUser);

            var manhattanMethod = AuthenticationMethodFactory.GetAuthenticationMethod(
                Method.Manhattan, hUserMedian, udUserMedian, hUserProfile, udUserProfile);
            var manhattanMethodResult = await manhattanMethod.Authenticate(user.Login, n, authenticationH, authenticationDU, thresholds);
            ProcessMethod(Method.Manhattan, manhattanMethodResult, user.Login, testAuthenticationsRecord.IsLegalUser);

            var filteredManhattanMethod = AuthenticationMethodFactory.GetAuthenticationMethod(
                Method.FilteredManhattan, hUserMedian, udUserMedian, hUserProfile, udUserProfile);
            var filteredManhattanMethodResult = await filteredManhattanMethod.Authenticate(user.Login, n, authenticationH, authenticationDU, thresholds);
            ProcessMethod(Method.FilteredManhattan, filteredManhattanMethodResult, user.Login, testAuthenticationsRecord.IsLegalUser);

            var scaledManhattanMethod = AuthenticationMethodFactory.GetAuthenticationMethod(
                Method.ScaledManhattan, hUserMedian, udUserMedian, hUserProfile, udUserProfile);
            var scaledManhattanMethodResult = await scaledManhattanMethod.Authenticate(user.Login, n, authenticationH, authenticationDU, thresholds);
            ProcessMethod(Method.ScaledManhattan, scaledManhattanMethodResult, user.Login, testAuthenticationsRecord.IsLegalUser);

            var ITADMethod = AuthenticationMethodFactory.GetAuthenticationMethod(
                Method.ITAD, hUserMedian, udUserMedian, hUserProfile, udUserProfile);
            var ITADMethodResult = await ITADMethod.Authenticate(user.Login, n, authenticationH, authenticationDU, thresholds);
            ProcessMethod(Method.ITAD, ITADMethodResult, user.Login, testAuthenticationsRecord.IsLegalUser);

            // --- ML Methods ---
            var cnnMethod = AuthenticationMethodFactory.GetAuthenticationMethod(
                Method.CNN, hUserMedian, udUserMedian, hUserProfile, udUserProfile);
            var cnnMethodResult = await cnnMethod.Authenticate(user.Login, n, authenticationH, authenticationDU, thresholds);
            ProcessMethod(Method.CNN, cnnMethodResult, user.Login, testAuthenticationsRecord.IsLegalUser);

            var gruMethod = AuthenticationMethodFactory.GetAuthenticationMethod(
                Method.GRU, hUserMedian, udUserMedian, hUserProfile, udUserProfile);
            var gruMethodResult = await gruMethod.Authenticate(user.Login, n, authenticationH, authenticationDU, thresholds);
            ProcessMethod(Method.GRU, gruMethodResult, user.Login, testAuthenticationsRecord.IsLegalUser);

            var knnMethod = AuthenticationMethodFactory.GetAuthenticationMethod(
                Method.KNN, hUserMedian, udUserMedian, hUserProfile, udUserProfile);
            var knnMethodResult = await knnMethod.Authenticate(user.Login, n, authenticationH, authenticationDU, thresholds);
            ProcessMethod(Method.KNN, knnMethodResult, user.Login, testAuthenticationsRecord.IsLegalUser);

            var lstmMethod = AuthenticationMethodFactory.GetAuthenticationMethod(
                Method.LSTM, hUserMedian, udUserMedian, hUserProfile, udUserProfile);
            var lstmMethodResult = await lstmMethod.Authenticate(user.Login, n, authenticationH, authenticationDU, thresholds);
            ProcessMethod(Method.LSTM, lstmMethodResult, user.Login, testAuthenticationsRecord.IsLegalUser);

            var mlpMethod = AuthenticationMethodFactory.GetAuthenticationMethod(
                Method.MLP, hUserMedian, udUserMedian, hUserProfile, udUserProfile);
            var mlpMethodResult = await mlpMethod.Authenticate(user.Login, n, authenticationH, authenticationDU, thresholds);
            ProcessMethod(Method.MLP, mlpMethodResult, user.Login, testAuthenticationsRecord.IsLegalUser);

            var naiveBayesMethod = AuthenticationMethodFactory.GetAuthenticationMethod(
                Method.NaiveBayes, hUserMedian, udUserMedian, hUserProfile, udUserProfile);
            var naiveBayesMethodResult = await naiveBayesMethod.Authenticate(user.Login, n, authenticationH, authenticationDU, thresholds);
            ProcessMethod(Method.NaiveBayes, naiveBayesMethodResult, user.Login, testAuthenticationsRecord.IsLegalUser);

            var randomForestMethod = AuthenticationMethodFactory.GetAuthenticationMethod(
                Method.RandomForest, hUserMedian, udUserMedian, hUserProfile, udUserProfile);
            var randomForestMethodResult = await randomForestMethod.Authenticate(user.Login, n, authenticationH, authenticationDU, thresholds);
            ProcessMethod(Method.RandomForest, randomForestMethodResult, user.Login, testAuthenticationsRecord.IsLegalUser);

            var svmMethod = AuthenticationMethodFactory.GetAuthenticationMethod(
                Method.SVM, hUserMedian, udUserMedian, hUserProfile, udUserProfile);
            var svmMethodResult = await svmMethod.Authenticate(user.Login, n, authenticationH, authenticationDU, thresholds);
            ProcessMethod(Method.SVM, svmMethodResult, user.Login, testAuthenticationsRecord.IsLegalUser);

            var xgboostMethod = AuthenticationMethodFactory.GetAuthenticationMethod(
                Method.XGBoost, hUserMedian, udUserMedian, hUserProfile, udUserProfile);
            var xgboostMethodResult = await xgboostMethod.Authenticate(user.Login, n, authenticationH, authenticationDU, thresholds);
            ProcessMethod(Method.XGBoost, xgboostMethodResult, user.Login, testAuthenticationsRecord.IsLegalUser);
        }

        private void WriteResultsToCsv(List<AuthenticationValidationResult> results, string saveDirectory) // Unchanged
        {
            if (results.Count == 0)
            {
                return;
            }

            string folderPath = Path.Combine(saveDirectory, $"N_{results[0].N}");
            if (!Directory.Exists(folderPath))
            {
                Directory.CreateDirectory(folderPath);
            }
            string filePath = Path.Combine(folderPath, $"{results[0].AuthenticationMethod}_results.csv");

            var records = results.Select(r => new
            {
                r.Login,
                r.IsLegalUser,
                AuthenticationMethod = r.AuthenticationMethod.ToString(),
                threshold = r.Threshold,
                r.IsAuthenticated,
                r.N,
                H_Score = r.DataResults.GetValueOrDefault(AuthenticationCalculationDataType.H),
                DU_Score = r.DataResults.GetValueOrDefault(AuthenticationCalculationDataType.UD),
                UU_Score = r.DataResults.GetValueOrDefault(AuthenticationCalculationDataType.UU),
                DD_Score = r.DataResults.GetValueOrDefault(AuthenticationCalculationDataType.DD),
                r.TotalAuthenticationScore
            });

            using (var writer = new StreamWriter(filePath))
            using (var csv = new CsvWriter(writer, CultureInfo.InvariantCulture))
            {
                Console.WriteLine($"Write in file {filePath}");
                csv.WriteRecords(records);
            }
        }
    }
}
