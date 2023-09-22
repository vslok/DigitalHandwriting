using System;
using System.Windows;
using DigitalHandwriting.Stores;
using DigitalHandwriting.ViewModels;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;

namespace DigitalHandwriting
{
    /// <summary>
    /// Interaction logic for App.xaml
    /// </summary>
    public partial class App : Application
    {
        public static IHost? AppHost { get; private set; }

        public App()
        {
            AppHost = Host.CreateDefaultBuilder()
            .ConfigureServices((hostContext, services) =>
            {
                services.AddSingleton<NavigationStore>();

                services.AddSingleton<MainViewModel>();
                services.AddSingleton<HomeViewModel>();
                services.AddSingleton<UserInfoViewModel>();
                services.AddTransient<RegistrationViewModel>();

                services.AddSingleton<MainWindow>((provider) =>
                {
                    return new MainWindow()
                    {
                        DataContext = provider.GetRequiredService<MainViewModel>()
                    };
                });

                services.AddTransient<Func<HomeViewModel>>((provider) =>
                {
                    return new Func<HomeViewModel>(
                        () => new HomeViewModel(
                            provider.GetRequiredService<Func<RegistrationViewModel>>(),
                            provider.GetRequiredService<NavigationStore>()
                            )
                    );
                });
                services.AddTransient<Func<RegistrationViewModel>>((provider) =>
                {
                    return new Func<RegistrationViewModel>(
                        () => new RegistrationViewModel(
                            provider.GetRequiredService<Func<HomeViewModel>>(),
                            provider.GetRequiredService<NavigationStore>())
                    );
                });
                //services.AddTransient<Func<UserInfoViewModel>>((provider) =>
                //{
                //    return new Func<UserInfoViewModel>(
                //        () => new UserInfoViewModel()
                //    );
                //});
            }).Build();
        }

        protected override async void OnStartup(StartupEventArgs e)
        {
            await AppHost!.StartAsync();

            var navigationStore = AppHost.Services.GetRequiredService<NavigationStore>();
            navigationStore.CurrentViewModel = AppHost.Services.GetRequiredService<HomeViewModel>();

            var mainWindow = AppHost.Services.GetRequiredService<MainWindow>();
            mainWindow.Show();

            base.OnStartup(e);
        }
    }
}
