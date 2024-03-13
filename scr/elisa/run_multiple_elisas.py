from elisa_compile import ELISACOMPILE
import tracemalloc

tracemalloc.start()

if __name__ == '__main__':
    # setting tab-delimited file pointing to excel files
    # ebov_fl = '/Users/gorkalasso/Documents/projects/Elisa_platform/covid19/JMD_excels/COVID_files.txt'
    # ebov_fl_0 = '/Users/gorkalasso/Dropbox (EinsteinMed)/Elisa_platform/ebolavirus_NPC1/input_files.txt'
    # ebov_fl_1 = '/Users/gorkalasso/Dropbox (EinsteinMed)/Elisa_platform/ebolavirus_NPC1/1_07_02_2020/npc1_input_07_02_2020.txt'
    # ebov_fl_2 = '/Users/gorkalasso/Dropbox (EinsteinMed)/Elisa_platform/ebolavirus_NPC1/2_01_12_2021/npc1_input_01_12_2021.txt'
    #ebov_fl_3 = "/Users/gorkalasso/Dropbox (EinsteinMed)/Elisa_platform/ebolavirus_NPC1/3_02_09_2021/npc1_input_02_09_2021.txt"
    # ebov_fl_4 = "/Users/gorkalasso/Dropbox (EinsteinMed)/Elisa_platform/ebolavirus_NPC1/3_02_09_2021/npc1_input_05_04_2021.txt"
    # ebov_fl_5 = "/Users/gorkalasso/Dropbox (EinsteinMed)/Elisa_platform/ebolavirus_NPC1/4_05_04_2021/npc1_LLOV_MLAV_05_04_2021.txt"
    # ebov_fl_6 = "/Users/gorkalasso/Dropbox (EinsteinMed)/Elisa_platform/ebolavirus_NPC1/5_06_06_2021/all_so_far_06_06_2021.txt"
    # ebov_fl_7 = "/Users/gorkalasso/Dropbox (EinsteinMed)/Elisa_platform/ebolavirus_NPC1/6_07_21_2021/cluster1_07_21_2021.txt"
    # ebov_fl_8 = "/Users/gorkalasso/Dropbox (EinsteinMed)/Elisa_platform/ebolavirus_NPC1/7_mlav_7_21_21/mlav_repeats.txt"
    # ebov_fl_9 = "/Users/gorkalasso/Dropbox (EinsteinMed)/Elisa_platform/ebolavirus_NPC1/8_new_clusters_8_5_2021/clusters_8_5_2021.txt"
    # ebov_fl_10 = "/Users/gorkalasso/Dropbox (EinsteinMed)/Elisa_platform/ebolavirus_NPC1/9_all_sortingClusters_8_5_2021/9_all_8_5_2021.txt"
    #longitudinal_fl = '/Users/gorkalasso/Dropbox (EinsteinMed)/Elisa_platform/covid19_longitudinal_Erika/covid10_longitudinal_input_08_13_2020.txt'

    # megan_fl = '/Users/gorkalasso/Dropbox (EinsteinMed)/Elisa_platform/megan_pcdh1_gngc/megan_competitionElisa_input.txt'
    # megan_fl = '/Users/gorkalasso/Dropbox (EinsteinMed)/Elisa_platform/megan_pcdh1_gngc/megan_neut_input.txt'
    # megan_fl = '/Users/gl2411/Dropbox (EinsteinMed)/Elisa_platform/megan_pcdh1_gngc/10_27_21/megan_competitionElisa_input.txt'
    # megan_fl2 = '/Users/gl2411/Dropbox (EinsteinMed)/Elisa_platform/megan_pcdh1_gngc/10_27_21/megan_competitionElisa__input_useWT2norm.txt'
    # megan_fl = '/Users/gl2411/Dropbox (EinsteinMed)/Elisa_platform/megan_pcdh1_gngc/10_27_21/megan_neut_input.txt'
    # megan_fl2 = '/Users/gl2411/Dropbox (EinsteinMed)/Elisa_platform/megan_pcdh1_gngc/10_27_21/megan_neut_input_useWT2norm.txt'
    #rct_segments_lai_fl = '/Users/gorkalasso/Dropbox (EinsteinMed)/Elisa_platform/rct/lai_lab/segments/rct_lai_segments.txt'
    # rct_samples_lai_fl1 = '/Users/gorkalasso/Dropbox (EinsteinMed)/Elisa_platform/rct/lai_lab/samples/rct_lai_samples_set1_to_3_fixed_blank.txt'
    rct_samples_lai_fl2 = '/Users/gorkalasso/Dropbox (EinsteinMed)/Elisa_platform/rct/lai_lab/samples_plus1day/rct_lai_samplesplusone_set6.txt'
    # rct_segments_pirofski_fl1 = '/Users/gorkalasso/Dropbox (EinsteinMed)/Elisa_platform/rct/pirofski_lab/segments/rct_pirofski_segments.txt'
    # rct_segments_pirofski_fl2 = '/Users/gorkalasso/Dropbox (EinsteinMed)/Elisa_platform/rct/pirofski_lab/segments/rct_pirofski_noNorm_segments.txt'
    # rct_segments_pirofski_set1 = '/Users/gorkalasso/Dropbox (EinsteinMed)/Elisa_platform/rct/pirofski_lab/segments/rct_pirofski_set1.txt'
    #q2_1_blind = "/Users/gorkalasso/Dropbox (EinsteinMed)/Elisa_platform/q2_neutralization/q2_blind_03_01_2021.txt"
    # q2_1_unblind = "/Users/gorkalasso/Dropbox (EinsteinMed)/Elisa_platform/q2_neutralization/q2_unblind_03_01_2021.txt"
    # q2_shipment3_fl = "/Users/gorkalasso/Dropbox (EinsteinMed)/Elisa_platform/rct/q2_neutralization/q2_shipment_1_to_3.txt"
    #q2_special_fl = "/Users/gorkalasso/Dropbox (EinsteinMed)/Elisa_platform/rct/q2_neutralization/q2_shipment_special.txt"
    #q2_shipment4_fl = "/Users/gl2411/Dropbox (EinsteinMed)/Elisa_platform/rct/q2_neutralization/q2_shipment_4.txt"
    # q2_fl = "/Users/gorkalasso/Dropbox (EinsteinMed)/Elisa_platform/rct/q2_neutralization/q2_shipment_5.txt"
    # q2_fl = "/Users/gorkalasso/Dropbox (EinsteinMed)/Elisa_platform/rct/q2_neutralization/q2_shipment_6_missing.txt"
    q2_fl = "/Users/gorkalasso/Dropbox (EinsteinMed)/Elisa_platform/rct/q2_neut/q2_neut_2023/q2_shipment_1.txt"
    # levi_1 = "/Users/gorkalasso/Dropbox (EinsteinMed)/Elisa_platform/rct/pirofski_lab/Levi_IGs_manual/IgM/igM_levi_set1.txt"
    # levi_2 = "/Users/gorkalasso/Dropbox (EinsteinMed)/Elisa_platform/rct/pirofski_lab/Levi_IGs_manual/IgA/igA_levi_set1.txt"
    pox_neut_eev = "/Users/gorkalasso/Dropbox (EinsteinMed)/Elisa_platform/pox_vaccine_response/vacv_neut_2_all.txt"
    github_example = "/Users/gorkalasso/Dropbox (EinsteinMed)/Elisa_platform/ebolavirus_NPC1/10_github_test/10_github.txt"
    elisa_comp = ELISACOMPILE(github_example)
    elisa_comp.read_infl()