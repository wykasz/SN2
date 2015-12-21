using Encog;
using Encog.Engine.Network.Activation;
using Encog.ML.Data;
using Encog.ML.Data.Basic;
using Encog.ML.Train;
using Encog.Neural.Networks;
using Encog.Neural.Networks.Layers;
using Encog.Neural.Networks.Training.Propagation.Resilient;
using Encog.Neural.Networks.Training.Propagation.Back;

using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;

namespace SN2
{
    class Program
    {
        static int basev = 32;
        static List<byte[]> getSet(string[] paths)
        {
            List<byte[]> L1 = new List<byte[]>();
            foreach (string s in paths)
            {
                Image img = Image.FromFile(s);
                Bitmap b = new Bitmap(img);
                byte[] byteArray = new byte[32 * 32];
                for (int i = 0; i < 32; i++)
                {
                    for (int j = 0; j < 32; j++)
                    {
                        byteArray[i * 32 + j] = b.GetPixel(i, j).R;
                    }
                }
                L1.Add(byteArray);
            }
            return L1;
        }

        static double[][] getTrSet(List<byte[]> L1, List<byte[]> L2)
        {
            double[][] ret = new double[L1.Count + L2.Count][];
            for (int i = 0; i < L1.Count; i++)
            {
                ret[i] = new double[basev * basev];
                for (int j = 0; j < basev * basev; j++)
                    ret[i][j] = L1[i][j];
            }
            for (int i = 0; i < L2.Count; i++)
            {
                ret[L1.Count+i] = new double[basev * basev];
                for (int j = 0; j < basev * basev; j++)
                    ret[L1.Count + i][j] = L2[i][j];
            }
            return ret;
        }

        static double[][] getIdeal(int s, double v, int s2, double v2)
        {
            double[][] ret = new double[s+s2][];
            for (int i = 0; i < s; i++)
            {
                ret[i] = new double[1];
                ret[i][0] = v;
            }
            for (int i = 0; i < s2; i++)
            {
                ret[s+i] = new double[1];
                ret[s+i][0] = v2;
            }
            return ret;
        }

        static BasicNetwork getNetwork()
        {
            var network = new BasicNetwork();
            network.AddLayer(new BasicLayer(new ActivationSigmoid(), true, basev*basev));
            network.AddLayer(new BasicLayer(new ActivationSigmoid(), false, 1));
            network.Structure.FinalizeStructure();

            network.Reset();
            return network;
        }

        static void Tarin(BasicNetwork N, IMLDataSet S)
        {
            int minErrIt = 0;
            double minErr = double.MaxValue;
            BasicNetwork best = new BasicNetwork();
            IMLTrain train = new Backpropagation(N, S);
            int epoch = 0;
            int max = 3000;
            do
            {
                train.Iteration();
                Console.WriteLine(@"Epoch #" + epoch + @" E r r o r : " + train.Error);
                epoch++;
                               
                /*double testerr = 0;
                foreach (IMLDataPair pair in S)
                {
                    IMLData output = N.Compute(pair.Input);
                    testerr += Math.Pow(output[0] - pair.Ideal[0], 2);
                }
                testerr = Math.Sqrt(testerr);
                Console.WriteLine(@"Epoch #" + epoch + @" E r r o r : " + train.Error);
                */
            } while (epoch < max);
        }

        static void Test(BasicNetwork N, IMLDataSet S, List<string> names)
        {
            double testerr = 0;
            double cerr = 0;
            int limit = 10;
            int i = 0;
            int errcount = 0;
            foreach (IMLDataPair pair in S)
            {
                //if (i >= limit) break;
                IMLData output = N.Compute(pair.Input);
                cerr += Math.Pow(output[0] - pair.Ideal[0], 2);
                if (Math.Abs(output[0] - pair.Ideal[0]) > 0.5)
                {
                    Console.WriteLine("Bad: "+names[i]);
                    errcount++;
                }
                else
                {
                    Console.WriteLine("Ok: " + names[i]);
                }
                testerr += cerr;
                i++;
            }
            testerr = Math.Sqrt(testerr);
            Console.WriteLine(" E r r o r : " + testerr);
            Console.WriteLine("Efficiency: " + (1.0 - (double)errcount / (double)i));
                
        }

        static void Ask(BasicNetwork N, IMLDataSet S, List<string> names)
        {
            Console.WriteLine("Ask");
            int i = 0;
            foreach (IMLDataPair pair in S)
            {
                IMLData output = N.Compute(pair.Input);
                if (output[0] > 0.5)
                {
                    Console.WriteLine("Las: " + names[i]);
                }
                else
                {
                    Console.WriteLine("Nie: " + names[i]);
                }
                i++;
            }
        }
        static void Main(string[] args)
        {
            string set_ok_path = "C:\\Users\\rwk\\Documents\\sn2\\data_clean_fft_ok_s\\google_ok";
            string set_nok_path = "C:\\Users\\rwk\\Documents\\sn2\\data_clean_fft_ok_s\\google_nok";
            string set_t_ok_path = "C:\\Users\\rwk\\Documents\\sn2\\data_clean_fft_ok_s\\gov_ok";
            string set_t_nok_path = "C:\\Users\\rwk\\Documents\\sn2\\data_clean_fft_ok_s\\gov_nok";
            string set_q_path = "C:\\Users\\rwk\\Documents\\sn2\\data_clean_fft_ok_s\\kontrolna";

            string[] set_ok = Directory.GetFiles(set_ok_path);
            string[] set_nok = Directory.GetFiles(set_nok_path);
            string[] set_t_ok = Directory.GetFiles(set_t_ok_path);
            string[] set_t_nok = Directory.GetFiles(set_t_nok_path);
            string[] set_q = Directory.GetFiles(set_q_path);

            List<string> L1 = new List<string>(set_ok);
            List<string> L2 = new List<string>(set_nok);
            foreach(string s in L2) 
                L1.Add(s);
            List<string> Lt1 = new List<string>(set_t_ok);
            List<string> Lt2 = new List<string>(set_t_nok);
            foreach (string s in Lt2)
                Lt1.Add(s);
            List<string> Lq = new List<string>(set_q);


            List<byte[]> L_ok = getSet(set_ok);
            List<byte[]> L_nok = getSet(set_nok);
            List<byte[]> L_t_ok = getSet(set_t_ok);
            List<byte[]> L_t_nok = getSet(set_t_nok);
            List<byte[]> L_q = getSet(set_q);
            List<byte[]> L_q2 = new List<byte[]>();

            double[][] d_set = getTrSet(L_ok, L_nok);
            double[][] id_set = getIdeal(L_ok.Count, 1, L_nok.Count, 0);
            double[][] d_t_set = getTrSet(L_t_ok, L_t_nok);
            double[][] id_t_set = getIdeal(L_t_ok.Count, 1, L_t_nok.Count, 0);
            double[][] d_q = getTrSet(L_q, L_q2);
            double[][] id_q = getIdeal(L_q.Count, 0, 0, 0);

            IMLDataSet trainingSet = new BasicMLDataSet(d_set, id_set);
            IMLDataSet testSet = new BasicMLDataSet(d_t_set, id_t_set);
            IMLDataSet questionSet = new BasicMLDataSet(d_q, id_q);

            Console.WriteLine("Case 1");

            BasicNetwork N1 = getNetwork();
            Tarin(N1, trainingSet);
            Test(N1, testSet, Lt1);

            Console.WriteLine("Case 2");
            BasicNetwork N2 = getNetwork();
            Tarin(N2, testSet);
            Test(N2, trainingSet, L1);
            //Ask(N, questionSet, Lq);
            System.Console.WriteLine(L_ok.Count);
        }
    }
}
