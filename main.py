import SimulationStudy
import Application
import ApplicationValidation

if __name__ == '__main__':

    """ This module contains results for two parts using the simulation data and application data."""
    # TODO: Uncomment below sections to achieve objectives.

    simul = SimulationStudy.SimulationStudy()
    app = Application.Application()
    app_val = ApplicationValidation.ApplicationValidation()

    if __name__ == '__main__':

        methods = ['logit', 'lgbm_c', 'decision_tree', 'rfc', 'svm_lin']
        source = 'peerj_submission'

        smote_status = [True, False]

        # source is the folder name in the local desktop

        """ To run simulation study"""

        n_trials = 1
        weights = [[0.5, 0.5], [0.7, 0.3], [0.9, 0.1]]
        sample_sizes = [1000]

        # for weight in weights:
        #     simul.main(analysis='pclfs-ext', source='', id_col='ID_col', target_col='Churn', status='new',
        #                methods=methods, smote_status=smote_status, n_features=30, n_redundant=0, n_repeated=0,
        #                n_classes=2, n_clusters_per_class=1, n_samples=sample_sizes, weight=weight, n_trials=n_trials,
        #                plot_true=True, csv=False, no_pc=2, step=1)

        """ To run the application"""

        app_val.main(analysis='pclfs-ext', source='peerj_submission', csv_name="SPECTF", id_col='id',
                 target_col='OVERALL_DIAGNOSIS', status='old', methods=methods, smote=smote_status, no_pc=2, weights=[],
                 n_features_to_select=None, no_of_sub_sets=None, plot_true=True, trial=1, n_informative=0, step=1)

        """ To validate the application"""

        # app_val.main(analysis='pclfs-ext', source='validate', csv_name="SPECTF", id_col='id',
        #          target_col='OVERALL_DIAGNOSIS', status='old', methods=methods, smote=smote_status, no_pc=2, weights=[],
        #          n_features_to_select=None, no_of_sub_sets=None, plot_true=False, trial=1, n_informative=0, step=1)

        # app.main(analysis='pclfs-ext', source='validate', csv_name="german", id_col='id',
        #          target_col='Customer_Type', status='old', methods=methods, smote=smote_status, no_pc=2, weights=[],
        #          n_features_to_select=None, no_of_sub_sets=None, plot_true=False, trial=1, n_informative=0, step=1)

        """Breast Cancer Wisconsin (Diagnostic)"""
        # app.main(analysis='pclfs-ext', source='validate', csv_name="wdbc", id_col='id',
        #          target_col='Diagnosis', status='old', methods=methods, smote=smote_status, no_pc=2, weights=[],
        #          n_features_to_select=None, no_of_sub_sets=None, plot_true=False, trial=1, n_informative=0, step=1)

        """Statlog (German Credit Data)"""
        # app.main(analysis='pclfs-ext', source='validate', csv_name="Hill_Valley_with_noise", id_col='id',
        #                  target_col='class', status='old', methods=methods, smote=smote_status, no_pc=2, weights=[],
        #                  n_features_to_select=None, no_of_sub_sets=None, plot_true=False, trial=1, n_informative=0, step=1)




        """ To validate the application"""
        # #
        # app_val.main(analysis='pclfs-ext', source='validate_new', csv_name="SPECTF", id_col='id',
        #              target_col='OVERALL_DIAGNOSIS', status='new', methods=methods, smote=smote_status, no_pc=2,
        #              weights=[],
        #              n_features_to_select=None, no_of_sub_sets=None, plot_true=False, trial=50, n_informative=0, step=1)

        # app_val.main(analysis='pclfs-ext', source='validate_new', csv_name="wdbc", id_col='id',
        #              target_col='Diagnosis', status='new', methods=methods, smote=smote_status, no_pc=2,
        #              weights=[],
        #              n_features_to_select=None, no_of_sub_sets=None, plot_true=False, trial=50, n_informative=0, step=1)

        # app_val.main(analysis='pclfs-ext', source='validate_new', csv_name="german", id_col='id',
        #              target_col='Customer_Type', status='new', methods=methods, smote=smote_status, no_pc=2,
        #              weights=[],
        #              n_features_to_select=None, no_of_sub_sets=None, plot_true=False, trial=50, n_informative=0, step=1)

        # app_val.main(analysis='pclfs-ext', source='validate_new', csv_name="ionosphere", id_col='id',
        #              target_col='target', status='new', methods=methods, smote=smote_status, no_pc=2,
        #              weights=[],
        #              n_features_to_select=None, no_of_sub_sets=None, plot_true=False, trial=50, n_informative=0, step=1)

        # app_val.main(analysis='pclfs-ext', source='validate_new', csv_name="sonar", id_col='id',
        #              target_col='target', status='new', methods=methods, smote=smote_status, no_pc=2,
        #              weights=[],
        #              n_features_to_select=None, no_of_sub_sets=None, plot_true=False, trial=50, n_informative=0, step=1)

        # app_val.main(analysis='pclfs-ext', source='validate_new', csv_name="madelon", id_col='id',
        #              target_col='class', status='new', methods=methods, smote=smote_status, no_pc=2,
        #              weights=[],
        #              n_features_to_select=None, no_of_sub_sets=None, plot_true=False, trial=50, n_informative=0, step=1)

        # app_val.main(analysis='pclfs-ext', source='validate_new', csv_name="musk", id_col='id',
        #              target_col='class', status='new', methods=methods, smote=smote_status, no_pc=2,
        #              weights=[],
        #              n_features_to_select=None, no_of_sub_sets=None, plot_true=False, trial=50, n_informative=0, step=1)
