import javax.imageio.ImageTranscoder;
import java.util.Arrays;

public class LunarLanderAgentBase {
    // The resolution of the observation space
    // The four variables of the observation space, from left to right:
    //   0: X component of the vector pointing to the middle of the platform from the lander
    //   1: Y component of the vector pointing to the middle of the platform from the lander
    //   2: X component of the velocity vector of the lander
    //   3: Y component of the velocity vector of the lander
    static final int[] OBSERVATION_SPACE_RESOLUTION = {15, 10, 14, 14}; // TODO

    final double[][] observationSpace;
    double[][][][][] qTable;
    final int[] envActionSpace;
    private final int nIterations;

    double epsilon = 1.0;
    int iteration = 0;
    boolean test = false;

    // your variables here
    // ...
    double[][][][][] bestTable;
    double bestReward = -200;
    double lastReward = -200;

    double alpha = 0.1; //1
    double gamma = 0.9; //0.7
    int epsilon_step = 100;
    double epsilon_decay = 0.9;  //0.75
    int save_interval = 1000;

    int epoch = 0;

    public LunarLanderAgentBase(double[][] observationSpace, int[] actionSpace, int nIterations) {
        this.observationSpace = observationSpace;
        this.qTable =
                new double[OBSERVATION_SPACE_RESOLUTION[0]]
                        [OBSERVATION_SPACE_RESOLUTION[1]]
                        [OBSERVATION_SPACE_RESOLUTION[2]]
                        [OBSERVATION_SPACE_RESOLUTION[3]]
                        [actionSpace.length];
        this.envActionSpace = actionSpace;
        this.nIterations = nIterations;
    }

    public static int[] quantizeState(double[][] observationSpace, double[] state) {
        int targetX = (int) ((state[0]+observationSpace[0][1])/((observationSpace[0][1]-observationSpace[0][0])/OBSERVATION_SPACE_RESOLUTION[0]));
        if(targetX == OBSERVATION_SPACE_RESOLUTION[0])
            targetX -= 1;

        int targetY = (int) (state[1]/(observationSpace[1][1]/OBSERVATION_SPACE_RESOLUTION[1]));
        if(targetY == OBSERVATION_SPACE_RESOLUTION[1])
            targetY -= 1;

        int velocityX = (int) (state[2]+observationSpace[2][1]/((observationSpace[2][1]-observationSpace[2][0])/OBSERVATION_SPACE_RESOLUTION[2]));
        if(velocityX == OBSERVATION_SPACE_RESOLUTION[2])
            velocityX -= 1;

        int velocityY = (int) (state[3]+observationSpace[3][1]/((observationSpace[3][1]-observationSpace[3][0])/OBSERVATION_SPACE_RESOLUTION[3]));
        if(velocityY == OBSERVATION_SPACE_RESOLUTION[3])
            velocityY -= 1;
        return new int[]{targetX,targetY,velocityX,velocityY};
    }

    public void epochEnd(double epochRewardSum) {
        return; // TODO
    }

    public int argMax(double[] newState){
        double max = newState[0];
        int re = 0;
        for (int i = 1; i < envActionSpace.length; i++) {
            if (newState[i] > max) {
                max = newState[i];
                re = i;
            }
        }
        return re;
    }

    public void learn(double[] oldState, int action, double[] newState, double reward) {
        int[] qOld = quantizeState(observationSpace, oldState);
        int[] qNew = quantizeState(observationSpace, newState);

        double qValueOld = qTable[qOld[0]][qOld[1]][qOld[2]][qOld[3]][action];
        int bestIndex = argMax(qTable[qNew[0]][qNew[1]][qNew[2]][qNew[3]]);

        double qValueBest  = qTable[qNew[0]][qNew[1]][qNew[2]][qNew[3]][bestIndex];

        double res = qValueOld + alpha * (reward + gamma * qValueBest - qValueOld);

        qTable[qOld[0]][qOld[1]][qOld[2]][qOld[3]][action] = res;

        iteration++;
        epsilon *= 0.99999;
    }

    public void trainEnd() {
        //qTable = null; // TODO
        test = true;
    }
}
