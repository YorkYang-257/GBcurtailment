# -*- coding: utf-8 -*-

import os
import click
import pandas as pd

from .utils import read_unit_commitment, read_model
from .model import build_model

import numpy as np

np.seterr(all='raise')

SOLVER = os.getenv('PSST_SOLVER', 'cbc')


@click.group()
@click.version_option('0.1.0', '--version')
def cli():
    pass


@cli.command()
@click.option('--data', default=None, type=click.Path(), help='Path to model data')
@click.option('--output', default='./xfertoames.dat', type=click.Path(), help='Path to output file')
@click.option('--solver', default=SOLVER, help='Solver')
def scuc(data, output, solver):
    click.echo("Running SCUC using PSST")

    c = read_model(data.strip("'"))
    #print(c.load)
    model = build_model(c)
    #model.solve(solver=solver)
    SolverOutcomes = model.solve(solver=solver)
    Status = str(SolverOutcomes[1])
    click.echo("Model for DAM combined SCUC/SCED is solved. Status: " + Status)
    with open(output, 'w') as outfile:

        instance = model._model
        print(instance.Demand['Bus1',2].value)
        #print(list(instance.GeneratorsAtBus['Bus3']))
        print(instance.posLoadGenerateMismatch['Bus1',2].value)
        print(sum(instance.StageCost[st].value for st in instance.StageSet))
        # l=sum(instance.LinePower[l,2].value for l in instance.LinesTo['Bus1']) \
        #        - sum(instance.LinePower[l,2].value for l in instance.LinesFrom['Bus1'])
        # print(l)
        #print(instance.LoadMismatchPenalty.value)
        #print(result.noload_cost)
        results = {}
        for g in instance.Generators.value:
            for t in instance.TimePeriods:
                results[(g, t)] = instance.UnitOn[g, t]

        for g in sorted(instance.Generators.value):
            outfile.write("%s\n" % str(g).ljust(8))
            for t in sorted(instance.TimePeriods):
                outfile.write("% 1d \n" % (int(results[(g, t)].value + 0.5)))
                #outfile.write("\tPowerGenerated: {}\n".format(instance.PowerGenerated[g, t].value))


@cli.command()
@click.option('--uc', default=None, type=click.Path(), help='Path to unit commitment file')
@click.option('--data', default=None, type=click.Path(), help='Path to model data')
@click.option('--output', default='./output.dat', type=click.Path(), help='Path to output file')
@click.option('--solver', default=SOLVER, help='Solver')
def sced(uc, data, output, solver):

    click.echo("Running SCED using PSST")

    # TODO : Fixme
    uc_df = pd.DataFrame(read_unit_commitment(uc.strip("'")))

    c = read_model(data.strip("'"))
    c.gen_status = uc_df.astype(int)
    model = build_model(c)
    # for i in model._model.UnitOn.index_set():
    #     print(model._model.UnitOn[i].value)
    model.solve(solver=solver)
    for t in model._model.TimePeriods:
        print(sum(model._model.PowerGenerated[g,t].value for g in model._model.GeneratorsAtBus['Bus1']))
        print(model._model.Demand['Bus1',t].value)

    with open(output.strip("'"), 'w') as f:
        f.write("LMP\n")
        for h, r in model.results.lmp.iterrows():
            bn = 1
            for _, lmp in r.items():
                if lmp is None:
                    lmp = 0
                f.write(str(bn) + ' : ' + str(h + 1) +' : ' + str(lmp) +"\n")
                bn = bn + 1
        f.write("END_LMP\n")

        f.write("GenCoResults\n")
        instance = model._model
        for g in instance.Generators.value:
            f.write("%s\n" % str(g).ljust(8))
            for t in instance.TimePeriods:
                f.write("Hour: {}\n".format(str(t + 1)))
                f.write("\tPowerGenerated: {}\n".format(instance.PowerGenerated[g, t].value))
                f.write("\tProductionCost: {}\n".format(instance.ProductionCost[g, t].value))
                f.write("\tStartupCost: {}\n".format(instance.StartupCost[g, t].value))
                f.write("\tShutdownCost: {}\n".format(instance.ShutdownCost[g, t].value))
        f.write("END_GenCoResults\n")
        f.write("VOLTAGE_ANGLES\n")
        for bus in sorted(instance.Buses):
            for t in instance.TimePeriods:
                f.write('{} {} : {}\n'.format(str(bus), str(t + 1), str(instance.Angle[bus, t].value)))
        f.write("END_VOLTAGE_ANGLES\n")
        # Write out the Daily LMP
        f.write("DAILY_BRANCH_LMP\n")
        f.write("END_DAILY_BRANCH_LMP\n")
        # Write out the Daily Price Sensitive Demand
        f.write("DAILY_PRICE_SENSITIVE_DEMAND\n")
        f.write("END_DAILY_PRICE_SENSITIVE_DEMAND\n")
        # Write out which hour has a solution
        f.write("HAS_SOLUTION\n")
        h = 0
        max_hour = 24  # FIXME: Hard-coded number of hours.
        while h < max_hour:
            f.write("1\t")  # FIXME: Hard-coded every hour has a solution.
            h += 1
        f.write("\nEND_HAS_SOLUTION\n")


@cli.command()
@click.option('--fd', default=None, type=click.Path(), help='Path to unit commitment file')
@click.option('--rd', default=None, type=click.Path(), help='Path to model data')
@click.option('--gi', default=None, type=click.Path(), help='Path to output file')
@click.option('--output', default='./output.dat', type=click.Path(), help='Path to output file')
@click.option('--solver', default=SOLVER, help='Solver')
def scuced(fd,rd,gi,output,solver):
    
    click.echo("Running SCUC and SCED using PSST")
    fd=fd.strip("'")
    gi=gi.strip("'")
    rd=rd.strip("'")
    day=1
    columns = [f'GenCo{i}' for i in range(1, 83)]  # This creates a list from GenCo1 to GenCo82
    store = np.zeros((8760, 82))

    df_st = pd.DataFrame(store, columns=columns)
    "DAM"
    while day<366:
        starthour=day*24-24
        endhour=(day)*24-1
        if day==365: endhour=(day)*24-1
        with open(fd) as f:
            data = f.read()
            data=data.splitlines(keepends=True)
            day_data = [line for line in data if len(line.split())<2 or (line.split()[1].isdigit() and starthour+1 <= int(line.split()[1]) <= endhour+1)]
            day_data.insert(0, data[0])
            day_data=' '.join(day_data)
            #print(day_data)
        with open(gi) as file:
            gidata=file.read()
        with open('inputdata.dat','w') as f:
            f.write(gidata+day_data)
        c = read_model('inputdata.dat')
        #print(c.load)
        model = build_model(c)
        #model.solve(solver=solver)
        SolverOutcomes = model.solve(solver=solver)
        Status = str(SolverOutcomes[1])
        click.echo("Model for DAM combined SCUC is solved. Status: " + Status)
        with open('uc.dat', 'w') as outfile:
    
            instance = model._model
            results = {}
            for g in instance.Generators.value:
                for t in instance.TimePeriods:
                # for t in range(24):
                    results[(g, t)] = instance.UnitOn[g, t]
    
            for g in sorted(instance.Generators.value):
                outfile.write("%s\n" % str(g).ljust(8))
                for t in sorted(instance.TimePeriods):
                    outfile.write("% 1d \n" % (int(results[(g, t)].value + 0.5)))
        
        
        "RTM"
        #endhour=(day)*24-1
        with open(rd) as f:
            data = f.read()
            data=data.splitlines(keepends=True)
            day_data = [line for line in data if len(line.split())<2 or (line.split()[1].isdigit() and starthour+1 <= int(line.split()[1]) <= endhour+1)]
            day_data.insert(0, data[0])
            day_data=' '.join(day_data)
        with open(gi) as file:
            gidata=file.read()
        with open('inputdata.dat','w') as f:
            f.write(gidata+day_data)
        uc='uc.dat'
        uc_df = pd.DataFrame(read_unit_commitment(uc.strip("'")))
        c = read_model('inputdata.dat')
        c.gen_status = uc_df.astype(int)
        model = build_model(c)
        #model.solve(solver=solver)
        SolverOutcomes = model.solve(solver=solver)
        Status = str(SolverOutcomes[1])
        click.echo("Model for RTM combined SCED is solved. Status: " + Status)
        # with open(output.strip("'"), 'w') as f:
        #     instance = model._model
        #     for g in instance.Generators.value:
        #         f.write("%s\n" % str(g).ljust(8))
        #         for t in instance.TimePeriods:
        #             f.write("Hour: {}\n".format(str(t + 1)))
        #             f.write("\tPowerGenerated: {}\n".format(instance.PowerGenerated[g, t].value))
        #             f.write("\tUnitOn: {}\n".format(instance.UnitOn[g, t].value))
        for i in model._model.Generators.value:
            for t in model._model.TimePeriods:
                df_st.loc[starthour+t,i]=model._model.PowerGenerated[i,t].value

        
        READ = False
        ll=[]
        DIRECTIVE = 'param: PowerGeneratedT0 UnitOnT0State MinimumPowerOutput MaximumPowerOutput MinimumUpTime MinimumDownTime NominalRampUpLimit NominalRampDownLimit StartupRampLimit ShutdownRampLimit ColdStartCost HotStartCost ShutdownCostCoefficient :=\n'
        for l in gidata.splitlines(keepends=True):
            if l.strip() == ';':
                READ = False
    
            if l == DIRECTIVE:
                READ = True
                ll.append(l)
                continue
    
            if READ is True:
                a=l.split()
                # a[1]=str(model._model.PowerGenerated[a[0],47].value)
                # a[2]=str(int(model._model.UnitOn[a[0],47].value))
                a[3]=str(int(model._model.ito[a[0]].value))
                a[4]=str(int(model._model.itf[a[0]].value))
                a[1]=str(model._model.PowerGenerated[a[0],23].value)
                a[2]=str(int(model._model.UnitOn[a[0],23].value))
                l=' '.join(a)
                l=l+'\n'
                #print(l)
            ll.append(l)
        newdata=''.join(ll)
                
    
        # totalgeneration=[]
        # for i in range(30):
        #     a=sum(model._model.PowerGenerated[g,i].value for g in model._model.Generators.value)
        #     totalgeneration.append(a)
        # print(totalgeneration)
        with open('gtest.dat','w') as f:
            f.write(newdata)
        day=day+1
        print(day)
        df_st.to_csv('g.csv', index=True) 
    #df_st.to_csv('g.csv', index=True) 
    
    return None



if __name__ == "__main__":
    cli()
