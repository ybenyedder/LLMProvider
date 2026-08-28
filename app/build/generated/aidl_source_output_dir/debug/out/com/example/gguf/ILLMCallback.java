/*
 * This file is auto-generated.  DO NOT MODIFY.
 */
package com.example.gguf;
public interface ILLMCallback extends android.os.IInterface
{
  /** Default implementation for ILLMCallback. */
  public static class Default implements com.example.gguf.ILLMCallback
  {
    /** Called for each generated token in real-time. */
    @Override public void onTokenReceived(java.lang.String token) throws android.os.RemoteException
    {
    }
    /** Called when the full generation is complete. */
    @Override public void onGenerationComplete(java.lang.String fullText) throws android.os.RemoteException
    {
    }
    @Override
    public android.os.IBinder asBinder() {
      return null;
    }
  }
  /** Local-side IPC implementation stub class. */
  public static abstract class Stub extends android.os.Binder implements com.example.gguf.ILLMCallback
  {
    /** Construct the stub at attach it to the interface. */
    public Stub()
    {
      this.attachInterface(this, DESCRIPTOR);
    }
    /**
     * Cast an IBinder object into an com.example.gguf.ILLMCallback interface,
     * generating a proxy if needed.
     */
    public static com.example.gguf.ILLMCallback asInterface(android.os.IBinder obj)
    {
      if ((obj==null)) {
        return null;
      }
      android.os.IInterface iin = obj.queryLocalInterface(DESCRIPTOR);
      if (((iin!=null)&&(iin instanceof com.example.gguf.ILLMCallback))) {
        return ((com.example.gguf.ILLMCallback)iin);
      }
      return new com.example.gguf.ILLMCallback.Stub.Proxy(obj);
    }
    @Override public android.os.IBinder asBinder()
    {
      return this;
    }
    @Override public boolean onTransact(int code, android.os.Parcel data, android.os.Parcel reply, int flags) throws android.os.RemoteException
    {
      java.lang.String descriptor = DESCRIPTOR;
      if (code >= android.os.IBinder.FIRST_CALL_TRANSACTION && code <= android.os.IBinder.LAST_CALL_TRANSACTION) {
        data.enforceInterface(descriptor);
      }
      switch (code)
      {
        case INTERFACE_TRANSACTION:
        {
          reply.writeString(descriptor);
          return true;
        }
      }
      switch (code)
      {
        case TRANSACTION_onTokenReceived:
        {
          java.lang.String _arg0;
          _arg0 = data.readString();
          this.onTokenReceived(_arg0);
          break;
        }
        case TRANSACTION_onGenerationComplete:
        {
          java.lang.String _arg0;
          _arg0 = data.readString();
          this.onGenerationComplete(_arg0);
          break;
        }
        default:
        {
          return super.onTransact(code, data, reply, flags);
        }
      }
      return true;
    }
    private static class Proxy implements com.example.gguf.ILLMCallback
    {
      private android.os.IBinder mRemote;
      Proxy(android.os.IBinder remote)
      {
        mRemote = remote;
      }
      @Override public android.os.IBinder asBinder()
      {
        return mRemote;
      }
      public java.lang.String getInterfaceDescriptor()
      {
        return DESCRIPTOR;
      }
      /** Called for each generated token in real-time. */
      @Override public void onTokenReceived(java.lang.String token) throws android.os.RemoteException
      {
        android.os.Parcel _data = android.os.Parcel.obtain();
        try {
          _data.writeInterfaceToken(DESCRIPTOR);
          _data.writeString(token);
          boolean _status = mRemote.transact(Stub.TRANSACTION_onTokenReceived, _data, null, android.os.IBinder.FLAG_ONEWAY);
        }
        finally {
          _data.recycle();
        }
      }
      /** Called when the full generation is complete. */
      @Override public void onGenerationComplete(java.lang.String fullText) throws android.os.RemoteException
      {
        android.os.Parcel _data = android.os.Parcel.obtain();
        try {
          _data.writeInterfaceToken(DESCRIPTOR);
          _data.writeString(fullText);
          boolean _status = mRemote.transact(Stub.TRANSACTION_onGenerationComplete, _data, null, android.os.IBinder.FLAG_ONEWAY);
        }
        finally {
          _data.recycle();
        }
      }
    }
    static final int TRANSACTION_onTokenReceived = (android.os.IBinder.FIRST_CALL_TRANSACTION + 0);
    static final int TRANSACTION_onGenerationComplete = (android.os.IBinder.FIRST_CALL_TRANSACTION + 1);
  }
  public static final java.lang.String DESCRIPTOR = "com.example.gguf.ILLMCallback";
  /** Called for each generated token in real-time. */
  public void onTokenReceived(java.lang.String token) throws android.os.RemoteException;
  /** Called when the full generation is complete. */
  public void onGenerationComplete(java.lang.String fullText) throws android.os.RemoteException;
}
