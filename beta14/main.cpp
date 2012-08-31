#include <RAT/DSReader.hh>
#include <RAT/DS/Root.hh>
#include <RAT/DS/EV.hh>

#include <TFile.h>
#include <TH1F.h>
#include <TVector3.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

#include "pmtpos.h"

float get_beta14(float3 &fit_position, float3* hit_pos, int nhits);

int main(int argc, char* argv[]) {
    TFile f("betas_e.root", "recreate");
    f.cd();

    TH1F hbeta14("hbeta14", "hbeta14", 200, 194.0, 196.0);

    const unsigned jobs = 10;
    for (unsigned jobnum=0; jobnum<jobs; jobnum++) {
        char filename[100];
        snprintf(filename, 100, "data/e_%i.root", jobnum);
        RAT::DSReader r(filename);
        unsigned count = 0;
        while(RAT::DS::Root* ds = r.NextEvent()) {
            for (int iev=0; iev<1; iev++) {
                RAT::DS::EV* ev = ds->GetEV(iev);
                printf("Processing job %u, event %u (GTID %x)\n", jobnum, count, ev->GetEventID());

                // get fit position
                TVector3 v = ev->GetFitResult("scintFitter").GetVertex(0).GetPosition();
                float3 fit_pos = make_float3((float)v[0], (float)v[1], (float)v[2]);

                // load positions of hit pmts
                int nhits = ev->GetNhits();
                float3* hit_pos = (float3*) malloc(nhits * sizeof(float3));
                for (size_t ipmt=0; ipmt<nhits; ipmt++) {
                    int pmtid = ev->GetPMTUnCal(ipmt)->GetID();
                    hit_pos[ipmt] = make_float3(pmtpos::x[pmtid], pmtpos::y[pmtid], pmtpos::z[pmtid]);
                }
                float beta14 = get_beta14(fit_pos, hit_pos, nhits);

                printf("nhit: %i, beta14: %1.12f\n", nhits, beta14);

                hbeta14.Fill(beta14);

                count++;
            }
        }
        hbeta14.Write();
    }

    f.Close();

    return 0;
}

