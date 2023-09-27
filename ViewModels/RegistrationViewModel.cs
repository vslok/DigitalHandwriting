using DigitalHandwriting.Commands;
using DigitalHandwriting.Context;
using DigitalHandwriting.Helpers;
using DigitalHandwriting.Models;
using DigitalHandwriting.Services;
using DigitalHandwriting.Stores;
using Microsoft.EntityFrameworkCore;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text.Json;
using System.Windows.Input;

namespace DigitalHandwriting.ViewModels
{
    public class RegistrationViewModel : BaseViewModel
    {
        public ICommand OnCheckTextBoxKeyDownEventCommand { get; set; }
        public ICommand OnCheckTextBoxKeyUpEventCommand { get; set; }
        public ICommand OnResetRegistrationWindowButtonClickCommand { get; set; }
        public ICommand OnRegistrationFinalizeButtonClickCommand { get; set; }
        public ICommand ReturnHomeCommand { get; set; }


        private int _checkTextCurrentLetterIndex = 0;

        private int _registrationStep = 0;

        private bool _isLoginBoxEnabled = true;

        private bool _isCheckTextBoxEnabled = false;

        private bool _isFinalizeButtonVisible = false;

        private string _checkTextWithUpperCase => ConstStrings.CheckText.ToUpper();

        private string _userCheckText = "";

        private string _userLogin = "";

        private string _userPassword = "";

        private ApplicationContext db = new ApplicationContext();

        private readonly KeyboardMetricsCollectionService _keyboardMetricsCollector;

        public RegistrationViewModel(
            Func<HomeViewModel> homeViewModelFactory,
            NavigationStore navigationStore,
            KeyboardMetricsCollectionService keyboardMetricsCollector)
        {
            OnCheckTextBoxKeyDownEventCommand = new RelayCommand<object>(OnCheckTextBoxKeyDownEvent);
            OnCheckTextBoxKeyUpEventCommand = new RelayCommand<object>(OnCheckTextBoxKeyUpEvent);
            OnRegistrationFinalizeButtonClickCommand = new NavigateCommand<HomeViewModel>(
                new NavigationService<HomeViewModel>(
                    navigationStore,
                    () => homeViewModelFactory()), OnRegistrationFinalizeButtonClick);
            ReturnHomeCommand = new NavigateCommand<HomeViewModel>(
                new NavigationService<HomeViewModel>(
                    navigationStore,
                    () => homeViewModelFactory()));
            OnResetRegistrationWindowButtonClickCommand = new Command(OnResetRegistrationWindowButtonClick);

            _keyboardMetricsCollector = keyboardMetricsCollector;

            db.Database.EnsureCreated();
            db.Users.Load();
        }

        public bool IsLoginBoxEnabled
        {
            get => _isLoginBoxEnabled;
            set => SetProperty(ref _isLoginBoxEnabled, value);
        }

        public bool IsCheckTextBoxEnabled
        {
            get => _isCheckTextBoxEnabled;
            set => SetProperty(ref _isCheckTextBoxEnabled, value);
        }

        public bool IsFinalizeButtonVisible
        {
            get => _isFinalizeButtonVisible;
            set => SetProperty(ref _isFinalizeButtonVisible, value);
        }

        public int RegistrationStep
        {
            get => _registrationStep;
            set
            {
                SetProperty(ref _registrationStep, value);
                if (value == 3)
                {
                    IsCheckTextBoxEnabled = false;
                    InvokeOnPropertyChangedEvent(nameof(IsPasswordTextBoxVisible));
                }
            }
        }

        public string UserLogin
        {
            get => _userLogin;
            set
            {
                SetProperty(ref _userLogin, value);
                IsCheckTextBoxEnabled = !string.IsNullOrEmpty(value);
            }
        }

        public bool IsPasswordTextBoxVisible => !IsCheckTextBoxEnabled && RegistrationStep != 0;

        public string UserPassword
        {
            get => _userPassword;
            set
            {
                SetProperty(ref _userPassword, value);
                if (_registrationStep == 3 && _userPassword.Length > 8)
                {
                    IsFinalizeButtonVisible = true;
                }
            }
        }

        public string UserCheckText
        {
            get => _userCheckText;
            set 
            {
                IsLoginBoxEnabled = false;
                SetProperty(ref _userCheckText, value);
            }
        }

        private void OnRegistrationFinalizeButtonClick()
        {
            User user = new User()
            {
                Login = UserLogin,
                KeyPressedTimes = JsonSerializer.Serialize(_keyboardMetricsCollector.GetKeyPressedTimesMedians()),
                BetweenKeysTimes = JsonSerializer.Serialize(_keyboardMetricsCollector.GetBetweenKeysTimesMedians()),
                Password = EncryptionService.GetPasswordHash(UserPassword, out string salt),
                Salt = salt
            };

            db.Users.Add(user);
            db.SaveChanges();
        }

        private void OnCheckTextBoxKeyUpEvent(object props)
        {
            var e = (KeyEventArgs)props;

            Trace.WriteLine($"{e.Key} = keyUp");

            _keyboardMetricsCollector.OnKeyUpEvent(e);

            if (_checkTextCurrentLetterIndex == _checkTextWithUpperCase.Length && UserCheckText.Length == _checkTextWithUpperCase.Length)
            {
                _keyboardMetricsCollector.IncreaseMetricsCollectingStep(_checkTextWithUpperCase);
                RegistrationStep++;
                UserCheckText = "";
                _checkTextCurrentLetterIndex = 0;
            }
        }

        private void OnCheckTextBoxKeyDownEvent(object props)
        {
            var e = (KeyEventArgs)props;
            Trace.WriteLine($"{e.Key} = keyDown");
            if (!Helper.CheckCurrentLetterKeyPressed(e, _checkTextCurrentLetterIndex, _checkTextWithUpperCase) || _registrationStep == 3)
            {
                e.Handled = true;
                return;
            }

            _keyboardMetricsCollector.OnKeyDownEvent(e);
            _checkTextCurrentLetterIndex++;
        }

        private void OnResetRegistrationWindowButtonClick()
        {
            RegistrationStep = 0;
            UserLogin = "";
            UserCheckText = "";
            UserPassword = "";
            _checkTextCurrentLetterIndex = 0;
            _keyboardMetricsCollector.ResetMetricsCollection();
            IsLoginBoxEnabled = true;
            IsCheckTextBoxEnabled = false;
            IsFinalizeButtonVisible = false;
        }
    }
}
