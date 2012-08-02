from rat import ROOT, dsreader
import beta14

if __name__ == '__main__':
    h = {
        1: ROOT.TH1F('hbeta_1', 'hbeta_1', 100, 1, 2),
        2: ROOT.TH1F('hbeta_2', 'hbeta_2', 100, 3.5, 4.5),
        3: ROOT.TH1F('hbeta_3', 'hbeta_3', 100, 12.2, 13.2),
        4: ROOT.TH1F('hbeta_4', 'hbeta_4', 100, 47.9, 48.9),
        '14': ROOT.TH1F('hbeta14', 'hbeta14', 200, 194.0, 196.0)
    }

    f = ROOT.TFile('betas_e.root', 'recreate')
    f.cd()

    for jobnum in range(10):
        count = 0
        for ds in dsreader('e_%i.root' % jobnum):
            for ev in [ds.GetEV(n) for n in range(1)]:
                print 'Processing job %i, event %i (GTID %x)' % (jobnum, count, ev.GetEventID())
                betas = beta14.calculate_betas(ev)
                for l in betas:
                    if l in h:
                        h[l].Fill(betas[l])
                        h[l].Write()

                count += 1

    f.Close()

